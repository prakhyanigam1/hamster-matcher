import cv2
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os

HAMSTER_LABELS = {
    "67hamster.jpg": "person doing the 67 hand gesture",
    "bubbleinyourmouth_hamster.jpg": "puffed out cheeks silly face",
    "concerned_side-eye.jpg": "side eye suspicious concerned face",
    "cryinghamster.jpg": "crying sad tears streaming down face",
    "eating_hamster.png": "eating stuffing food in mouth",
    "eyebrow_raise.webp": "raised eyebrow skeptical judging face",
    "glasses_hamster.jpg": "wearing glasses nerdy smart face",
    "hand_heart_hamster.jpg": "making heart shape with hands",
    "happy_hamster.jpg": "big smile happy joyful face",
    "idk_hamster.jpg": "shrugging confused i dont know face",
    "kissing_hamster.webp": "kissing lips puckered up face",
    "shocked_hamster.jpg": "shocked wide eyes open mouth surprised",
    "silence_hamster.webp": "quiet shushing finger on lips",
    "sleeping_hamster.webp": "sleeping tired eyes closed",
    "smilinghamster.jpg": "smiling grinning happy face",
    "stronghamster.jpg": "flexing muscles strong tough face",
    "tongue_out.jpg": "sticking tongue out playful silly",
}

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

HAMSTER_DIR = "hamsters/hamster_dataset"

def get_embedding(image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.vision_model(**inputs)
        embedding = outputs.pooler_output
    return embedding / embedding.norm(dim=-1, keepdim=True)

print("loading hamsters...")
hamsters = {}
for fname in os.listdir(HAMSTER_DIR):
    if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        path = os.path.join(HAMSTER_DIR, fname)
        img = Image.open(path).convert("RGB")
        img_emb = get_embedding(img)
        hamsters[fname] = img_emb
print(f"loaded {len(hamsters)} hamsters")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    user_embedding = get_embedding(pil_img)

    best_match = None
    best_score = -1
    for name, emb in hamsters.items():
        score = (user_embedding @ emb.T).item()
        if score > best_score:
            best_score = score
            best_match = name

    hamster_path = os.path.join(HAMSTER_DIR, best_match)
    hamster_img = cv2.imread(hamster_path)
    hamster_img = cv2.resize(hamster_img, (frame.shape[1], frame.shape[0]))

    label = f"you are: {best_match} ({best_score:.2f})"
    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    combined = cv2.hconcat([frame, hamster_img])
    cv2.imshow("hamster matcher", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()