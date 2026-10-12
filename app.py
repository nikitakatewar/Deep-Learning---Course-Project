import gradio as gr
import time
import cv2
import os
from models.fused_model import FusedDetectorInference

print("Loading Omni-Detect V6 Model (this may take a few seconds)...")
detector = FusedDetectorInference("weights/best_model.pth", n_frames=16)
print("Model loaded successfully!")

def analyze_media(video_path, image_path):
    if not video_path and not image_path:
        return "Please upload a video or an image.", {"Error": 1.0}
        
    start_time = time.time()
    
    target_path = video_path
    temp_video = None
    
    # If it's an image, convert to a 16-frame temporary video
    if image_path and not video_path:
        img = cv2.imread(image_path)
        if img is None:
            return "Error reading image.", {"Error": 1.0}
            
        h, w, _ = img.shape
        temp_video = "temp_image_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, 30.0, (w, h))
        # Write 16 identical frames
        for _ in range(16):
            out.write(img)
        out.release()
        target_path = temp_video

    try:
        results = detector.predict(target_path)
    except Exception as e:
        if temp_video and os.path.exists(temp_video):
            os.remove(temp_video)
        return f"Error analyzing media: {str(e)}", {"Error": 1.0}
        
    if temp_video and os.path.exists(temp_video):
        os.remove(temp_video)
        
    calc_time = time.time() - start_time
    
    final_score = results.get("final_score", 0.0)
    face_score = results.get("face_score", 0.0)
    frame_score = results.get("frame_score", 0.0)
    temp_score = results.get("temporal_score", 0.0)
    
    verdict = "🚨 FAKE (AI-Generated/Manipulated) 🚨" if final_score > 0.50 else "✅ REAL (Authentic) ✅"
    
    media_type = "Image" if image_path and not video_path else "Video"
    
    report = f'''
### Verdict: {verdict}
**Overall Tampering Risk Score: {final_score*100:.1f}%**

#### Branch Analysis Breakdown:
- 🕵️‍♂️ **Branch A (Face Expert):** {face_score*100:.1f}% risk
- 🔭 **Branch B (Spectral Expert):** {frame_score*100:.1f}% risk
- ⏱️ **Branch C (Temporal Expert):** {temp_score*100:.1f}% risk *(Ignored for still {media_type}s)*

*Processing {media_type} took {calc_time:.1f} seconds.*
'''
    label_dict = {"FAKE": final_score, "REAL": 1.0 - final_score}
    return report, label_dict

with gr.Blocks() as demo:
    gr.Markdown("# 🛡️ Omni-Detect V6 Deepfake Scanner")
    gr.Markdown("Upload a **Video** OR an **Image** to analyze it using our 3-Branch Gated Fusion architecture.")
    
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("Upload Video"):
                    video_input = gr.Video(label="Video")
                with gr.TabItem("Upload Image"):
                    image_input = gr.Image(type="filepath", label="Image")
                
            analyze_btn = gr.Button("🔍 Scan Media", variant="primary")
        
        with gr.Column():
            label_output = gr.Label(label="Confidence")
            report_output = gr.Markdown(label="Detailed Report")
            
    analyze_btn.click(fn=analyze_media, inputs=[video_input, image_input], outputs=[report_output, label_output])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, inbrowser=True, theme=gr.themes.Soft())
