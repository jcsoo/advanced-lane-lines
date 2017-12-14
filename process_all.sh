#!/bin/sh

python pipeline.py project_video.mp4 out_video/pv_bw.mp4
python pipeline.py challenge_video.mp4 out_video/cv_bw.mp4
python pipeline.py harder_challenge_.mp4 out_video/hcv_bw.mp4