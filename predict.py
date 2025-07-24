# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
import numpy as np
from typing import Optional, List, Dict, Any
from PIL import Image
from diffusers import (
    LTXConditionPipeline,
    LTXLatentUpsamplePipeline,
)
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_video


MODEL_CACHE = "LTX-Video-0.9.8-13B-distilled"
UPSAMPLER_CACHE = "ltxv-spatial-upscaler-0.9.7"
MODEL_URL = "https://weights.replicate.delivery/default/Lightricks/LTX-Video-0.9.8-13B-distilled/model.tar"
UPSAMPLER_URL = "https://weights.replicate.delivery/default/Lightricks/ltxv-spatial-upscaler-0.9.7/model.tar"

# Optimized configuration parameters from HF Space
OPTIMIZATION_CONFIG = {
    "decode_timestep": 0.05,
    "image_cond_noise_scale": 0.15,
    "mixed_precision": True,
}


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Download main model weights
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
        if not os.path.exists(UPSAMPLER_CACHE):
            download_weights(UPSAMPLER_URL, UPSAMPLER_CACHE)

        print("Creating LTX Video pipeline with memory optimizations...")
        # Initialize main pipeline with optimizations
        self.pipe = LTXConditionPipeline.from_pretrained(
            MODEL_CACHE,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        
        # Initialize upsampler pipeline with optimizations
        self.pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(
            UPSAMPLER_CACHE,
            vae=self.pipe.vae,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        
        # Enable memory optimizations
        self.pipe.vae.enable_tiling()
        self.pipe.enable_attention_slicing()
        
        # Enable additional VAE optimizations from HF Space
        if hasattr(self.pipe.vae, 'enable_slicing'):
            self.pipe.vae.enable_slicing()
        
        # Additional memory optimizations
        if hasattr(self.pipe, 'enable_vae_slicing'):
            self.pipe.enable_vae_slicing()
        
        if hasattr(self.pipe, 'enable_vae_tiling'):
            self.pipe.enable_vae_tiling()

        # Ensure pipelines are on CUDA
        if not self.pipe.device.type == "cuda":
            print("Moving models to CUDA...")
            self.pipe.to("cuda")
            self.pipe_upsample.to("cuda")

        print("Model setup complete with HF Space optimizations enabled")

    def calculate_padding(self, actual_height, actual_width, padded_height, padded_width):
        """Calculate padding values for dimension adjustment"""
        pad_height = padded_height - actual_height
        pad_width = padded_width - actual_width
        
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        
        return (pad_left, pad_right, pad_top, pad_bottom)

    def round_to_nearest_resolution_acceptable_by_vae(self, height, width):
        """Round dimensions to nearest resolution acceptable by VAE"""
        height = height - (height % self.pipe.vae_spatial_compression_ratio)
        width = width - (width % self.pipe.vae_spatial_compression_ratio)
        return height, width

    def round_to_macro_block_size(self, height, width, block_size=16):
        """Round dimensions to nearest multiple of block_size for video codec compatibility"""
        # Round to nearest multiple of block_size
        rounded_height = round(height / block_size) * block_size
        rounded_width = round(width / block_size) * block_size
        
        # Ensure minimum size
        if rounded_height < block_size:
            rounded_height = block_size
        if rounded_width < block_size:
            rounded_width = block_size
            
        return rounded_height, rounded_width


    def calculate_segment_plan(self, max_duration_seconds: int, fps: int, num_frames_per_segment: int, overlap_frames: int) -> List[Dict[str, Any]]:
        """Calculate how to split the video into segments"""
        total_frames = max_duration_seconds * fps
        effective_frames_per_segment = num_frames_per_segment - overlap_frames
        
        segments = []
        current_output_frame = 0
        segment_index = 0
        
        while current_output_frame < total_frames:
            # Calculate how many frames this segment should produce
            remaining_frames = total_frames - current_output_frame
            frames_this_segment = min(effective_frames_per_segment, remaining_frames)
            
            # For the first segment, we generate all frames
            # For subsequent segments, we generate with overlap
            generate_frames = num_frames_per_segment if segment_index == 0 else num_frames_per_segment
            
            segments.append({
                "segment_index": segment_index,
                "output_start_frame": current_output_frame,
                "output_end_frame": current_output_frame + frames_this_segment,
                "generate_frames": generate_frames,
                "frames_to_use": frames_this_segment + (overlap_frames if segment_index > 0 else 0),
                "is_final_segment": current_output_frame + frames_this_segment >= total_frames
            })
            
            current_output_frame += frames_this_segment
            segment_index += 1
        
        return segments


    def create_conditioning_from_frames(self, frames: List[Image.Image], num_conditioning_frames: int = 8) -> List:
        """Create conditioning from the last frames of previous segment"""
        if not frames or len(frames) < num_conditioning_frames:
            return None
        
        # Use the last N frames as conditioning
        conditioning_frames = frames[-num_conditioning_frames:]
        
        # Convert to video format for conditioning
        video_from_frames = load_video(export_to_video(conditioning_frames))
        condition = LTXVideoCondition(video=video_from_frames, frame_index=0)
        return [condition]

    def blend_transition_frames(self, prev_frames: List[Image.Image], curr_frames: List[Image.Image], overlap_frames: int) -> List[Image.Image]:
        """Blend overlapping frames between segments for smooth transitions"""
        if not prev_frames or not curr_frames or overlap_frames <= 0:
            return curr_frames
        
        # Get the frames we want to blend
        prev_tail = prev_frames[-overlap_frames:] if len(prev_frames) >= overlap_frames else prev_frames
        curr_head = curr_frames[:overlap_frames] if len(curr_frames) >= overlap_frames else curr_frames
        
        blended_transition = []
        
        # Blend the overlapping frames
        num_blend_frames = min(len(prev_tail), len(curr_head), overlap_frames)
        for i in range(num_blend_frames):
            prev_frame = prev_tail[i]
            curr_frame = curr_head[i]
            
            # Calculate blend weight (linear transition from prev to curr)
            alpha = (i + 1) / (num_blend_frames + 1)
            
            # Blend frames
            prev_array = np.array(prev_frame)
            curr_array = np.array(curr_frame)
            blended_array = (1 - alpha) * prev_array + alpha * curr_array
            blended_frame = Image.fromarray(blended_array.astype(np.uint8))
            
            blended_transition.append(blended_frame)
        
        # Return only the non-overlapping frames from current segment
        result_frames = curr_frames[overlap_frames:] if len(curr_frames) > overlap_frames else []
        
        return blended_transition + result_frames

    def _generate_long_video(
        self,
        prompt: str,
        image: Optional[Path],
        video: Optional[Path],
        negative_prompt: str,
        resolution: int,
        aspect_ratio: str,
        num_frames: int,
        num_inference_steps: int,
        guidance_scale: float,
        fps: int,
        generator: torch.Generator,
        downscale_factor: float,
        denoise_strength: float,
        final_inference_steps: int,
        conditioning_frames: int,
        go_fast: bool,
        max_duration_seconds: int
    ) -> Path:
        """Generate long video using autoregressive approach"""
        print(f"Starting long video generation: {max_duration_seconds}s at {fps}fps")
        print(f"Using main prompt for all segments: {prompt[:100]}...")
        
        # Calculate segment plan with fixed 8-frame overlap
        segment_overlap_frames = 8
        segment_plan = self.calculate_segment_plan(max_duration_seconds, fps, num_frames, segment_overlap_frames)
        print(f"Generated {len(segment_plan)} segments")
        
        all_frames = []
        previous_segment_frames = None
        
        for segment_info in segment_plan:
            segment_idx = segment_info["segment_index"]
            start_time = segment_info["output_start_frame"] / fps
            end_time = segment_info["output_end_frame"] / fps
            
            print(f"Generating segment {segment_idx + 1}/{len(segment_plan)} ({start_time:.1f}s - {end_time:.1f}s)")
            
            # Determine conditioning for this segment
            conditions = None
            if segment_idx == 0:
                # First segment: use original image/video if provided
                if video is not None:
                    print("Using input video for first segment conditioning")
                    input_video = load_video(str(video))
                    conditioning_video = input_video[:conditioning_frames]
                    condition1 = LTXVideoCondition(video=conditioning_video, frame_index=0)
                    conditions = [condition1]
                elif image is not None:
                    print("Using input image for first segment conditioning")
                    pil_image = Image.open(str(image)).convert("RGB")
                    # We'll handle resizing in the segment generation
                    video_from_image = load_video(export_to_video([pil_image]))
                    condition1 = LTXVideoCondition(video=video_from_image, frame_index=0)
                    conditions = [condition1]
            else:
                # Subsequent segments: use frames from previous segment
                print(f"Using {segment_overlap_frames} frames from previous segment for conditioning")
                conditions = self.create_conditioning_from_frames(previous_segment_frames, segment_overlap_frames)
            
            # Generate this segment
            segment_frames = self._generate_single_segment(
                prompt=prompt,
                negative_prompt=negative_prompt,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                num_frames=segment_info["generate_frames"],
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                downscale_factor=downscale_factor,
                denoise_strength=denoise_strength,
                final_inference_steps=final_inference_steps,
                go_fast=go_fast,
                conditions=conditions,
                image=image if segment_idx == 0 else None  # Only use image for first segment
            )
            
            # Process frames for output
            if segment_idx == 0:
                # First segment: use all frames except those that will be overlapped
                frames_to_add = segment_frames[:segment_info["frames_to_use"]]
                all_frames.extend(frames_to_add)
            else:
                # Subsequent segments: blend transition and add new frames
                print(f"Blending {segment_overlap_frames} transition frames")
                
                # Use last overlap_frames from previous segment and first overlap_frames from current
                prev_tail = all_frames[-segment_overlap_frames:]
                curr_head = segment_frames[:segment_overlap_frames]
                curr_body = segment_frames[segment_overlap_frames:segment_info["frames_to_use"]]
                
                # Replace the tail of all_frames with blended transition
                all_frames = all_frames[:-segment_overlap_frames]  # Remove the frames to be blended
                
                # Create blended transition
                for i in range(segment_overlap_frames):
                    if i < len(prev_tail) and i < len(curr_head):
                        alpha = (i + 1) / (segment_overlap_frames + 1)
                        prev_array = np.array(prev_tail[i])
                        curr_array = np.array(curr_head[i])
                        blended_array = (1 - alpha) * prev_array + alpha * curr_array
                        blended_frame = Image.fromarray(blended_array.astype(np.uint8))
                        all_frames.append(blended_frame)
                
                # Add the new content from current segment
                all_frames.extend(curr_body)
            
            previous_segment_frames = segment_frames
            
            print(f"Segment {segment_idx + 1} complete. Total frames: {len(all_frames)}")
        
        # Export final video
        output_video_path_str = "/tmp/output.mp4"
        export_to_video(all_frames, output_video_path_str, fps=fps)
        
        print(f"Long video generation complete: {len(all_frames)} frames ({len(all_frames)/fps:.1f}s)")
        return Path(output_video_path_str)

    def _generate_single_segment(
        self,
        prompt: str,
        negative_prompt: str,
        resolution: int,
        aspect_ratio: str,
        num_frames: int,
        num_inference_steps: int,
        guidance_scale: float,
        generator: torch.Generator,
        downscale_factor: float,
        denoise_strength: float,
        final_inference_steps: int,
        go_fast: bool,
        conditions: Optional[List],
        image: Optional[Path] = None
    ) -> List[Image.Image]:
        """Generate a single video segment using the existing pipeline logic"""
        # Calculate dimensions (reusing logic from main predict function)
        height = resolution
        
        if aspect_ratio == "match_input_image" and image is not None:
            try:
                pil_image = Image.open(str(image)).convert("RGB")
                img_width, img_height = pil_image.size
                image_aspect_ratio = img_width / img_height
                width = int(height * image_aspect_ratio)
            except Exception:
                width = int(height * 16 / 9)
        elif aspect_ratio == "16:9":
            width = int(height * 16 / 9)
        elif aspect_ratio == "1:1":
            width = height
        elif aspect_ratio == "9:16":
            width = int(height * 9 / 16)
        else:
            width = int(height * 16 / 9)
        
        # Round dimensions to be divisible by 32
        original_height, original_width = height, width
        height = ((height - 1) // 32 + 1) * 32
        width = ((width - 1) // 32 + 1) * 32
        
        expected_height, expected_width = original_height, original_width
        padded_height, padded_width = height, width
        
        # Calculate padding
        padding_values = self.calculate_padding(expected_height, expected_width, padded_height, padded_width)
        
        # Handle image conditioning if provided for first segment
        if image is not None and conditions is None:
            pil_image = Image.open(str(image)).convert("RGB")
            pil_image = pil_image.resize((padded_width, padded_height), Image.LANCZOS)
            video_from_image = load_video(export_to_video([pil_image]))
            condition1 = LTXVideoCondition(video=video_from_image, frame_index=0)
            conditions = [condition1]
        
        # Prepare call kwargs
        call_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": padded_height,
            "width": padded_width,
            "num_frames": num_frames,
            "generator": generator,
            "conditions": conditions,
            "guidance_scale": guidance_scale,
            "decode_timestep": OPTIMIZATION_CONFIG["decode_timestep"],
            "image_cond_noise_scale": OPTIMIZATION_CONFIG["image_cond_noise_scale"],
            "output_type": "latent",
        }
        
        # Generate at downscaled resolution
        downscaled_height = int(padded_height * downscale_factor)
        downscaled_width = int(padded_width * downscale_factor)
        downscaled_height, downscaled_width = self.round_to_nearest_resolution_acceptable_by_vae(
            downscaled_height, downscaled_width
        )
        
        # Set inference steps based on fast mode
        if go_fast:
            call_kwargs["num_inference_steps"] = max(8, num_inference_steps // 2)
        else:
            call_kwargs["num_inference_steps"] = num_inference_steps
        
        # First pass
        first_pass_kwargs = call_kwargs.copy()
        first_pass_kwargs.update({
            "height": downscaled_height,
            "width": downscaled_width,
        })
        
        latents = self.pipe(**first_pass_kwargs).frames
        
        # Upscale
        upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
        upscaled_latents = self.pipe_upsample(
            latents=latents,
            output_type="latent"
        ).frames
        
        # Final denoising
        final_kwargs = call_kwargs.copy()
        final_kwargs.update({
            "height": upscaled_height,
            "width": upscaled_width,
            "latents": upscaled_latents,
            "denoise_strength": denoise_strength,
            "output_type": "pil",
        })
        
        if go_fast:
            final_kwargs["num_inference_steps"] = max(4, final_inference_steps // 2)
        else:
            final_kwargs["num_inference_steps"] = final_inference_steps
        
        video = self.pipe(**final_kwargs).frames[0]
        
        # Remove padding and resize
        if padding_values != (0, 0, 0, 0):
            pad_left, pad_right, pad_top, pad_bottom = padding_values
            video = [
                frame.crop((pad_left, pad_top, frame.width - pad_right, frame.height - pad_bottom))
                for frame in video
            ]
        
        if video[0].size != (expected_width, expected_height):
            video = [frame.resize((expected_width, expected_height), Image.LANCZOS) for frame in video]
        
        return video

    def predict(
        self,
        prompt: str = Input(description="Text prompt for video generation"),
        image: Optional[Path] = Input(
            description="Input image for image-to-video generation. If provided, will generate video from this image.",
            default=None
        ),
        video: Optional[Path] = Input(
            description="Input video for video-to-video generation. If provided, will generate video from this video. Takes precedence over image if both are provided.",
            default=None
        ),
        negative_prompt: str = Input(
            description="Negative prompt for video generation.",
            default="worst quality, inconsistent motion, blurry, jittery, distorted"
        ),
        resolution: int = Input(
            description="Resolution for the output video (height in pixels).",
            default=720,
            choices=[480, 720]
        ),
        aspect_ratio: str = Input(
            description="Aspect ratio for the output video.",
            default="match_input_image",
            choices=["16:9", "1:1", "9:16", "match_input_image"]
        ),
        num_frames: int = Input(
            description="Number of frames per segment. Use 257 for ~10.7s segments at 24fps (recommended for long videos).",
            default=121,
            ge=9,
            le=257
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps for initial generation.",
            default=24,
            ge=2,
            le=50
        ),
        guidance_scale: float = Input(
            description="Guidance scale. Recommended range: 3.0-3.5.",
            default=3.0,
            ge=1.0,
            le=10.0
        ),
        fps: int = Input(
            description="Frames per second for the output video.",
            default=24,
            ge=1,
            le=60
        ),
        seed: Optional[int] = Input(
            description="Random seed for reproducible results. Leave blank for a random seed.",
            default=None
        ),
        downscale_factor: float = Input(
            description="Factor to downscale initial generation (recommended: 2/3 for better quality).",
            default=0.667,
            ge=0.1,
            le=1.0
        ),
        denoise_strength: float = Input(
            description="Denoising strength for final refinement step.",
            default=0.4,
            ge=0.0,
            le=1.0
        ),
        final_inference_steps: int = Input(
            description="Number of inference steps for final denoising.",
            default=10,
            ge=1,
            le=50
        ),
        conditioning_frames: int = Input(
            description="Number of frames to use for video-to-video conditioning (only used when video input is provided).",
            default=21,
            ge=1,
            le=50
        ),
        go_fast: bool = Input(
            description="Enable fast mode with skip layer strategies (20-40% faster but slightly lower quality).",
            default=True
        ),
        max_duration_seconds: int = Input(
            description="Target video duration in seconds. Videos longer than 10s use autoregressive generation.",
            default=5,
            ge=5,
            le=60
        ),
    ) -> Path:
        """Generate a video from a text prompt, image+text prompt, or video+text prompt using LTX Video 0.9.8-dev with HF Space optimizations"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator().manual_seed(seed)

        # Determine if we need long video generation (autoregressive approach)
        enable_long_video = max_duration_seconds > 10
        
        # Validate parameters for long video generation
        if enable_long_video:
            # Ensure num_frames is suitable for segmentation
            if num_frames < 17:  # Minimum for meaningful segments with overlap
                raise ValueError("num_frames must be at least 17 for long video generation")
            
            # Use fixed overlap of 8 frames
            segment_overlap_frames = 8
            
            # Validate segment overlap
            if segment_overlap_frames >= num_frames:
                raise ValueError("segment_overlap_frames must be less than num_frames")
            
            # Validate max duration
            effective_frames_per_segment = num_frames - segment_overlap_frames
            if effective_frames_per_segment <= 0:
                raise ValueError("Invalid configuration: overlap frames too large for segment size")
            
            # Ensure minimum segment duration makes sense
            min_segment_duration = effective_frames_per_segment / fps
            if min_segment_duration < 2.0:
                print(f"Warning: Very short effective segment duration ({min_segment_duration:.1f}s). Consider increasing num_frames or reducing overlap.")
            
            print(f"Long video mode: {max_duration_seconds}s target, ~{effective_frames_per_segment} effective frames per segment")

        # Handle long video generation
        if enable_long_video:
            return self._generate_long_video(
                prompt=prompt,
                image=image,
                video=video,
                negative_prompt=negative_prompt,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                fps=fps,
                generator=generator,
                downscale_factor=downscale_factor,
                denoise_strength=denoise_strength,
                final_inference_steps=final_inference_steps,
                conditioning_frames=conditioning_frames,
                go_fast=go_fast,
                max_duration_seconds=max_duration_seconds
            )

        # Calculate width and height based on resolution and aspect ratio
        height = resolution
        
        # Handle match_input_image aspect ratio
        if aspect_ratio == "match_input_image":
            if image is not None:
                # Load image to get its dimensions
                try:
                    pil_image = Image.open(str(image)).convert("RGB")
                    img_width, img_height = pil_image.size
                    # Calculate aspect ratio from image
                    image_aspect_ratio = img_width / img_height
                    width = int(height * image_aspect_ratio)
                    print(f"Using aspect ratio from input image: {img_width}x{img_height} -> {width}x{height} (ratio: {image_aspect_ratio:.3f})")
                except Exception as e:
                    print(f"Warning: Could not read input image dimensions: {e}. Falling back to 16:9.")
                    width = int(height * 16 / 9)
                    aspect_ratio = "16:9"  # Update for logging
            else:
                print("Warning: match_input_image selected but no image provided. Using 16:9 aspect ratio.")
                width = int(height * 16 / 9)
                aspect_ratio = "16:9"  # Update for logging
        elif aspect_ratio == "16:9":
            width = int(height * 16 / 9)
        elif aspect_ratio == "1:1":
            width = height
        elif aspect_ratio == "9:16":
            width = int(height * 9 / 16)
        else:
            raise ValueError(f"Unsupported aspect ratio: {aspect_ratio}")
        
        # Round dimensions to be divisible by 32 for better compatibility
        original_height, original_width = height, width
        height = ((height - 1) // 32 + 1) * 32
        width = ((width - 1) // 32 + 1) * 32
        
        if original_height != height or original_width != width:
            print(f"Rounded dimensions for optimal processing: {original_width}x{original_height} -> {width}x{height}")
        else:
            print(f"Dimensions already optimal: {width}x{height}")
        
        print(f"Using resolution: {width}x{height} ({aspect_ratio})")

        expected_height, expected_width = original_height, original_width
        padded_height, padded_width = height, width
        
        # Calculate padding for dimension adjustment
        padding_values = self.calculate_padding(expected_height, expected_width, padded_height, padded_width)

        # Prepare conditions for image-to-video or video-to-video if inputs are provided
        conditions = None
        mode = "Text-to-Video"
        
        if video is not None:
            print("Loading and processing input video for video-to-video generation")
            try:
                # Load the input video
                input_video = load_video(str(video))
                # Use only the specified number of frames as conditioning
                conditioning_video = input_video[:conditioning_frames]
                # Create condition from the video (using frame 0)
                condition1 = LTXVideoCondition(video=conditioning_video, frame_index=0)
                conditions = [condition1]
                mode = "Video-to-Video"
                print(f"Video-to-video condition created successfully using {conditioning_frames} frames")
            except Exception as e:
                raise ValueError(f"Failed to process input video: {e}")
        elif image is not None:
            print("Loading and processing input image for image-to-video generation")
            try:
                # Load and process the input image
                pil_image = Image.open(str(image)).convert("RGB")
                
                # Resize image to padded dimensions for compatibility
                pil_image = pil_image.resize((padded_width, padded_height), Image.LANCZOS)
                
                # Convert image to video format as required by the model
                video_from_image = load_video(export_to_video([pil_image]))
                # Create condition from the video (using frame 0)
                condition1 = LTXVideoCondition(video=video_from_image, frame_index=0)
                conditions = [condition1]
                mode = "Image-to-Video"
                print("Image-to-video condition created successfully")
            except Exception as e:
                raise ValueError(f"Failed to process input image: {e}")

        # Prepare optimized call kwargs with HF Space optimizations
        call_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": padded_height,
            "width": padded_width,
            "num_frames": num_frames,
            "generator": generator,
            "conditions": conditions,
            "guidance_scale": guidance_scale,
            "decode_timestep": OPTIMIZATION_CONFIG["decode_timestep"],
            "image_cond_noise_scale": OPTIMIZATION_CONFIG["image_cond_noise_scale"],
            "output_type": "latent",
        }

        # Part 1. Generate video at smaller resolution with optimizations
        downscaled_height = int(padded_height * downscale_factor)
        downscaled_width = int(padded_width * downscale_factor)
        downscaled_height, downscaled_width = self.round_to_nearest_resolution_acceptable_by_vae(
            downscaled_height, downscaled_width
        )

        print(f"Step 1: {mode} generation at downscaled resolution: {downscaled_width}x{downscaled_height}")
        
        # Use optimized parameters for faster inference if fast mode is enabled
        if go_fast:
            print("Using fast mode with optimized parameters")
            # Use fewer inference steps for faster generation
            call_kwargs["num_inference_steps"] = max(8, num_inference_steps // 2)
            # Enable CFG rescaling for better quality with fewer steps
            if hasattr(self.pipe.scheduler, 'config') and hasattr(self.pipe.scheduler.config, 'rescale_betas_zero_snr'):
                self.pipe.scheduler.config.rescale_betas_zero_snr = True
        else:
            call_kwargs["num_inference_steps"] = num_inference_steps

        # First pass generation
        first_pass_kwargs = call_kwargs.copy()
        first_pass_kwargs.update({
            "height": downscaled_height,
            "width": downscaled_width,
        })
        
        latents = self.pipe(**first_pass_kwargs).frames

        # Part 2. Upscale generated video using latent upsampler
        upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
        print(f"Step 2: Upsampling to: {upscaled_width}x{upscaled_height}")
        upscaled_latents = self.pipe_upsample(
            latents=latents,
            output_type="latent"
        ).frames

        # Part 3. Final denoising with optimizations
        print(f"Step 3: Final denoising with {final_inference_steps} steps")
        
        final_kwargs = call_kwargs.copy()
        final_kwargs.update({
            "height": upscaled_height,
            "width": upscaled_width,
            "latents": upscaled_latents,
            "denoise_strength": denoise_strength,
            "output_type": "pil",
        })
        
        # Use optimized parameters for final pass if fast mode is enabled
        if go_fast:
            final_kwargs["num_inference_steps"] = max(4, final_inference_steps // 2)
        else:
            final_kwargs["num_inference_steps"] = final_inference_steps

        video = self.pipe(**final_kwargs).frames[0]

        # Part 4. Remove padding and resize to expected resolution
        if padding_values != (0, 0, 0, 0):
            pad_left, pad_right, pad_top, pad_bottom = padding_values
            # Remove padding from each frame
            video = [
                frame.crop((pad_left, pad_top, frame.width - pad_right, frame.height - pad_bottom))
                for frame in video
            ]

        # Final resize if needed
        if video[0].size != (expected_width, expected_height):
            print(f"Step 4: Final resize to: {expected_width}x{expected_height}")
            video = [frame.resize((expected_width, expected_height), Image.LANCZOS) for frame in video]

        # Define output path
        output_video_path_str = "/tmp/output.mp4"

        export_to_video(video, output_video_path_str, fps=fps)

        print(f"[+] {mode} generation complete: {output_video_path_str}")
        return Path(output_video_path_str)
