#!/bin/bash
# Social-media video (1280x720, H.264 + AAC) for an audio sample: title and subtitle on top, and the full waveform
# lighting up as playback advances, with a white playhead.
# Usage: tools/waveform_video.sh audio.wav "Title" "Subtitle" output.mp4
set -euo pipefail
AUDIO="$1"; TITLE="$2"; SUBTITLE="$3"; OUTPUT="$4"
BOLD_FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
REGULAR_FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
DURATION=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$AUDIO")
WAVEFORM_IMAGE=$(mktemp --suffix=.png)
trap 'rm -f "$WAVEFORM_IMAGE"' EXIT

# normalization only shapes the drawing; the video keeps the original sound
ffmpeg -hide_banner -loglevel error -y -i "$AUDIO" -filter_complex \
  "[0:a]dynaudnorm=f=150:g=7,showwavespic=s=1120x320:split_channels=0:colors=0x4FD1C5:scale=sqrt:draw=full" \
  -frames:v 1 "$WAVEFORM_IMAGE"

# the part not yet played is covered by a dark veil that slides with time
ffmpeg -hide_banner -loglevel error -y -i "$AUDIO" -loop 1 -framerate 30 -i "$WAVEFORM_IMAGE" -filter_complex "
color=c=0x0F172A:s=1280x720:r=30:d=$DURATION[background];
color=c=0x0F172A@0.75:s=1140x340:r=30:d=$DURATION,format=rgba[veil];
color=c=white@0.9:s=3x350:r=30:d=$DURATION,format=rgba[playhead];
[background][1:v]overlay=x=80:y=260:shortest=1[with_waveform];
[with_waveform][veil]overlay=x='80+1120*t/$DURATION':y=250:eval=frame[with_veil];
[with_veil][playhead]overlay=x='80+1120*t/$DURATION':y=245:eval=frame,
drawtext=fontfile=$BOLD_FONT:text='$TITLE':fontcolor=white:fontsize=58:x=(w-text_w)/2:y=90,
drawtext=fontfile=$REGULAR_FONT:text='$SUBTITLE':fontcolor=0x94A3B8:fontsize=30:x=(w-text_w)/2:y=170,
format=yuv420p[video]" -map "[video]" -map 0:a -c:v libx264 -preset medium -crf 20 -c:a aac -ar 48000 -b:a 160k \
  -shortest -movflags +faststart "$OUTPUT"
echo "$OUTPUT: $(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUTPUT") s"
