# LTX Video 0.9.7 Distilled - Replicate Cog

A high-quality video generation model that creates videos from text prompts, images, or existing videos using the LTX Video 0.9.7-distilled architecture.

## Features

- **Text-to-Video**: Generate videos from text descriptions
- **Image-to-Video**: Animate static images into videos  
- **Video-to-Video**: Transform existing videos with new prompts
- **Multiple aspect ratios**: 16:9, 1:1, 9:16, or match input image
- **Flexible resolution**: 480p or 720p output
- **Fast mode**: 20-40% faster generation with optimized settings

## Usage

### Text-to-Video
```bash
cog predict -i prompt="A cat playing with a ball of yarn"
```

### Image-to-Video
```bash
cog predict -i prompt="A cute little penguin takes out a book and starts reading it" -i image=@peng.png
```

### Video-to-Video
```bash
cog predict -i prompt="The video depicts a winding mountain road covered in snow, with a single vehicle traveling along it. The road is flanked by steep, rocky cliffs and sparse vegetation. The landscape is characterized by rugged terrain and a river visible in the distance. The scene captures the solitude and beauty of a winter drive through a mountainous region" -i video=@cosmos.mp4
```

## Key Parameters

- `prompt`: Text description of the desired video
- `resolution`: Output height in pixels (480 or 720)
- `aspect_ratio`: Video dimensions (16:9, 1:1, 9:16, or match_input_image)
- `num_frames`: Number of frames to generate (9-257, default: 121)
- `fps`: Frames per second for output video (default: 24)
- `go_fast`: Enable fast mode for quicker generation (default: true)

## Output

Returns an MP4 video file with the generated content at the specified resolution and frame rate.

## Model Info

- Based on LTX Video 0.9.7-distilled by Lightricks
- Includes spatial upsampler for enhanced quality
- Optimized for memory efficiency and fast inference
- Support for various conditioning modes 