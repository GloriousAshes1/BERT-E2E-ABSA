import argparse
import torch
import numpy as np
from transformers import BertConfig, BertTokenizer # Chỉ dùng Bert cho ví dụ này

# Import các lớp và hàm từ các tệp của bạn
from absa_layer import BertABSATagger # Giả sử bạn đang dùng BertABSATagger
from glue_utils import InputExample, convert_examples_to_seq_features, ABSAProcessor
from seq_utils import ot2bieos_ts, bio2ot_ts, tag2ts

# Định nghĩa MODEL_CLASSES (tương tự như trong work.py hoặc main.py)
# Nếu bạn dùng XLNet, hãy thêm cấu hình cho XLNet vào đây
MODEL_CLASSES = {
    'bert': (BertConfig, BertABSATagger, BertTokenizer),
}

def predict_single_sentence(args, sentence_text):
    """
    Dự đoán khía cạnh và tình cảm cho một câu đơn lẻ.

    Args:
        args (argparse.Namespace): Namespace chứa các đối số như
                                   model_type, model_checkpoint_path, tokenizer_path,
                                   max_seq_length, tagging_schema.
        sentence_text (str): Câu đầu vào cần dự đoán.
    """
    # 1. Xác định lớp mô hình và tokenizer
    _, model_class, tokenizer_class = MODEL_CLASSES[args.model_type.lower()]

    # 2. Tải mô hình và tokenizer đã huấn luyện
    print(f"Loading tokenizer from: {args.tokenizer_path}")
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path, do_lower_case=args.do_lower_case)

    print(f"Loading model from checkpoint: {args.model_checkpoint_path}")
    model = model_class.from_pretrained(args.model_checkpoint_path)

    # Lấy tagging_schema từ config của model đã lưu nếu có, nếu không thì dùng từ args
    # Điều này quan trọng để đảm bảo sử dụng đúng schema mà mô hình đã được huấn luyện.
    # Thông thường, training_args.bin được lưu cùng mô hình, chứa thông tin này.
    # Hoặc, config của mô hình (config.json) cũng có thể chứa nó.
    # Ở đây, chúng ta sẽ ưu tiên args.tagging_schema nếu người dùng cung cấp rõ ràng khi gọi hàm,
    # nhưng trong thực tế, bạn nên đảm bảo nó khớp với lúc huấn luyện.
    tagging_schema_from_model_config = getattr(model.config, 'tagging_schema', args.tagging_schema)
    if tagging_schema_from_model_config != args.tagging_schema:
        print(f"Warning: Tagging schema from model config ({tagging_schema_from_model_config}) "
              f"differs from provided args.tagging_schema ({args.tagging_schema}). "
              f"Using schema from model config: {tagging_schema_from_model_config}")
    current_tagging_schema = tagging_schema_from_model_config


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 3. Tiền xử lý câu đầu vào
    words_in_sentence = sentence_text.split(' ')
    # Tạo nhãn giả 'O' để tương thích với convert_examples_to_seq_features
    dummy_labels = ['O'] * len(words_in_sentence)
    example = InputExample(guid="predict-single", text_a=sentence_text, label=dummy_labels)
    examples = [example]

    # Lấy label_list (quan trọng cho việc ánh xạ ID nhãn)
    processor = ABSAProcessor()
    # Sử dụng tagging_schema mà mô hình đã được huấn luyện (lấy từ config hoặc args)
    label_list = processor.get_labels(tagging_schema=current_tagging_schema)
    label_map = {label: i for i, label in enumerate(label_list)}
    absa_id2tag = {i: label for i, label in enumerate(label_list)}


    # Chuyển đổi thành features
    # Các tham số như cls_token_at_end, pad_on_left, cls_token_segment_id, pad_token_segment_id
    # cần phải giống với lúc huấn luyện mô hình của bạn.
    # Thông thường, các giá trị này phụ thuộc vào args.model_type.
    # Ví dụ cho BERT:
    cls_token_at_end = False
    pad_on_left = False
    cls_token_segment_id = 0 # BERT thường dùng 0 cho [CLS] và segment A
    pad_token_segment_id = 0 # BERT thường dùng 0 cho padding

    if args.model_type == 'xlnet': # Ví dụ nếu là XLNet
        cls_token_at_end = True
        pad_on_left = True
        cls_token_segment_id = 2
        pad_token_segment_id = 4


    features = convert_examples_to_seq_features(
        examples=examples,
        label_list=label_list, # Danh sách các nhãn BIEOS
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length, # Sử dụng max_seq_length từ args
        cls_token_at_end=cls_token_at_end,
        cls_token=tokenizer.cls_token,
        sep_token=tokenizer.sep_token,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        sequence_a_segment_id=0, # Cho một câu đơn
        cls_token_segment_id=cls_token_segment_id,
        pad_on_left=pad_on_left,
        pad_token_segment_id=pad_token_segment_id,
        mask_padding_with_zero=True
    )

    # Trích xuất tensor đầu vào
    input_ids = torch.tensor([features[0].input_ids], dtype=torch.long).to(device) # Thêm batch dimension
    attention_mask = torch.tensor([features[0].input_mask], dtype=torch.long).to(device)
    token_type_ids = None
    if args.model_type in ['bert', 'xlnet']:
        token_type_ids = torch.tensor([features[0].segment_ids], dtype=torch.long).to(device)

    evaluate_label_ids_for_sentence = features[0].evaluate_label_ids

    # 4. Đưa dữ liệu vào mô hình để dự đoán
    with torch.no_grad():
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if token_type_ids is not None:
            inputs['token_type_ids'] = token_type_ids

        outputs = model(**inputs)
        logits = outputs[0] # BertABSATagger trả về logits ở vị trí đầu tiên

        predicted_label_ids_at_subword_level = []
        if model.tagger_config.absa_type != 'crf':
            preds_tensor = torch.argmax(logits, dim=-1)
            predicted_label_ids_at_subword_level = preds_tensor[0].cpu().numpy()
        else:
            predicted_tags_list = model.tagger.viterbi_tags(logits=logits, mask=attention_mask)
            predicted_label_ids_at_subword_level = predicted_tags_list[0] # Đây là list các ID

    # 5. Hậu xử lý dự đoán
    # Ánh xạ từ subword về word
    predicted_label_ids_at_word_level = [predicted_label_ids_at_subword_level[i] for i in evaluate_label_ids_for_sentence]

    # Chuyển ID thành nhãn chữ
    predicted_tags_at_word_level = [absa_id2tag[label_id] for label_id in predicted_label_ids_at_word_level]

    # Chuyển đổi lược đồ (nếu cần) và trích xuất TS
    # Giả định rằng tag2ts hoạt động tốt nhất với BIEOS, nên chúng ta chuẩn hóa về BIEOS
    final_tags_for_ts_extraction = []
    if current_tagging_schema == 'OT':
        final_tags_for_ts_extraction = ot2bieos_ts(predicted_tags_at_word_level)
    elif current_tagging_schema == 'BIO':
        temp_ot_tags = bio2ot_ts(predicted_tags_at_word_level)
        final_tags_for_ts_extraction = ot2bieos_ts(temp_ot_tags)
    elif current_tagging_schema == 'BIEOS':
        final_tags_for_ts_extraction = predicted_tags_at_word_level
    else:
        raise Exception(f"Unsupported tagging schema for postprocessing: {current_tagging_schema}")

    targeted_sentiments = tag2ts(ts_tag_sequence=final_tags_for_ts_extraction)

    # 6. Hiển thị kết quả
    output_ts_strings = []
    for t in targeted_sentiments:
        begin_idx, end_idx, sentiment_label = t
        aspect_words = words_in_sentence[begin_idx : end_idx + 1]
        aspect_string = " ".join(aspect_words)
        output_ts_strings.append(f"('{aspect_string}', {sentiment_label})")

    print(f"\n--- Dự đoán cho câu: '{sentence_text}' ---")
    if output_ts_strings:
        print(f"Các khía cạnh và tình cảm được phát hiện:")
        for item in output_ts_strings:
            print(f"  {item}")
    else:
        print("  Không phát hiện thấy khía cạnh-tình cảm nào.")
    print("--- Kết thúc dự đoán ---")
    return output_ts_strings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Các đối số này phải khớp với cách bạn đã huấn luyện mô hình
    parser.add_argument("--model_type", default="bert", type=str, help="Loại mô hình (ví dụ: bert)")
    parser.add_argument("--model_checkpoint_path", type=str, required=True,
                        help="Đường dẫn đến thư mục checkpoint của mô hình đã huấn luyện (ví dụ: ./output_dir/checkpoint-xxxx)")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Đường dẫn đến thư mục chứa tokenizer đã lưu (thường là thư mục output chính của huấn luyện)")
    parser.add_argument("--max_seq_length", default=128, type=int, help="Độ dài chuỗi tối đa")
    parser.add_argument("--tagging_schema", default="BIEOS", type=str,
                        help="Lược đồ gắn thẻ được sử dụng khi huấn luyện (OT, BIO, BIEOS)")
    parser.add_argument("--do_lower_case", action='store_true', help="Sử dụng tokenizer không phân biệt hoa thường (nếu mô hình là uncased)")
    # Bạn có thể thêm các đối số khác nếu cần, ví dụ như absa_type nếu nó ảnh hưởng đến việc load config

    cli_args = parser.parse_args()

    # Ví dụ câu cần dự đoán
    my_sentence_1 = "The food was absolutely wonderful, from preparation to presentation, very pleasing."
    my_sentence_2 = "The service is terrible and the price is too high."
    my_sentence_3 = "I love the new phone, its camera is amazing but the battery drains quickly."
    my_sentence_4 = "This place is okay, nothing special."


    predict_single_sentence(cli_args, my_sentence_1)
    predict_single_sentence(cli_args, my_sentence_2)
    predict_single_sentence(cli_args, my_sentence_3)
    predict_single_sentence(cli_args, my_sentence_4)

    # Ví dụ cách chạy từ dòng lệnh:
    # python your_prediction_script_name.py --model_checkpoint_path ./bert-bilstm_cnn-rest14-finetune/checkpoint-1200 --tokenizer_path ./bert-bilstm_cnn-rest14-finetune/ --tagging_schema BIEOS --do_lower_case
    # (Thay thế ./bert-bilstm_cnn-rest14-finetune/checkpoint-1200 và ./bert-bilstm_cnn-rest14-finetune/ bằng đường dẫn thực tế của bạn)