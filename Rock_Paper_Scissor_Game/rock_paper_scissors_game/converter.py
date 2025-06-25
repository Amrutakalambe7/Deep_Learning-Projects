from moviepy.editor import VideoFileClip

# Load your .avi file
clip = VideoFileClip("game_record.avi")

# Export it to .mp4
clip.write_videofile("stone_paper_scissor.mp4", codec="libx264")

