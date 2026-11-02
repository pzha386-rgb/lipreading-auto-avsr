import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from src.model.avhubert2text import AV2TextForConditionalGeneration
from src.dataset.load_data import load_feature
from transformers import Speech2TextTokenizer
import torch

if __name__ == "__main__":
    
    # Choose language to run example
    AVAILABEL_LANGUAGES = ["ar", "de", "el", "en", "es", "fr", "it", "pt", "ru", "multilingual"]
    language = "ru"
    assert language in AVAILABEL_LANGUAGES, f"Language {language} is not available, please choose one of {AVAILABEL_LANGUAGES}"
    
    
    # Load model and tokenizer
    model_name_or_path = f"nguyenvulebinh/AV-HuBERT-MuAViC-{language}"
    model = AV2TextForConditionalGeneration.from_pretrained(model_name_or_path, cache_dir='./model-bin')
    tokenizer = Speech2TextTokenizer.from_pretrained(model_name_or_path, cache_dir='./model-bin')
    
    model = model.cuda().eval()
    
    # Load example video and audio
    video_example = f"./example/video_processed/{language}_lip_movement.mp4"
    audio_example = f"./example/video_processed/{language}_audio.wav"
    if not os.path.exists(video_example) or not os.path.exists(audio_example):
        print(f"WARNING: Example video and audio for {language} is not available english will be used instead")
        video_example = f"./example/video_processed/en_lip_movement.mp4"
        audio_example = f"./example/video_processed/en_audio.wav"
    
    # Load and process example
    sample = load_feature(
        video_example,
        audio_example
    )
    
    audio_feats = sample['audio_source'].cuda()
    video_feats = sample['video_source'].cuda()
    attention_mask = torch.BoolTensor(audio_feats.size(0), audio_feats.size(-1)).fill_(False).cuda()
    
    # Generate text
    output = model.generate(
        audio_feats,
        attention_mask=attention_mask,
        video=video_feats,
        max_length=1024,
    )

    print(tokenizer.batch_decode(output, skip_special_tokens=True))