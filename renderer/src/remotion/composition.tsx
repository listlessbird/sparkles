import React, { useEffect, useState } from "react";
import {
  AbsoluteFill,
  Audio,
  Composition,
  Img,
  Sequence,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  Easing,
} from "remotion";
import { z } from "zod";
import { CaptionSchema } from "../schema";
import { defaultProps } from "./default";
import { loadFont } from "@remotion/google-fonts/Quicksand";
import { getAudioDurationInSeconds } from "@remotion/media-utils";

type VideoComponentProps = {
  topic: string;
  script: { textContent: string; imagePrompt: string }[];
  images: string[];
  speechUrl: string;
  captions: z.infer<typeof CaptionSchema>[];
};

const { fontFamily: fontBold } = loadFont("normal", { weights: ["700"] });
const { fontFamily: fontNormal } = loadFont("normal", { weights: ["500"] });

export const VideoComponent: React.FC<VideoComponentProps> = ({
  topic,
  script,
  images,
  speechUrl,
  captions,
}) => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();
  const [audioDuration, setAudioDuration] = useState<number | null>(null);

  useEffect(() => {
    const loadAudioDuration = async () => {
      if (speechUrl) {
        const duration = await getAudioDurationInSeconds(speechUrl);
        setAudioDuration(duration);
      }
    };
    loadAudioDuration();
  }, [speechUrl]);

  const getCurrentCaption = (frame: number) => {
    const timeInMs = (frame / fps) * 1000;
    return (
      captions.find(
        (caption) => timeInMs >= caption.start && timeInMs <= caption.end
      ) || null
    );
  };

  const getTotalDuration = () => {
    if (audioDuration) {
      return Math.max(
        captions[captions.length - 1].end,
        audioDuration * 1000,
        script.length * 5000
      );
    }
    return Math.max(captions[captions.length - 1].end, script.length * 5000);
  };

  const getDurationInFrames = () => {
    return Math.ceil((getTotalDuration() / 1000) * fps);
  };

  const getCurrentImageIndex = (frame: number) => {
    const timeInMs = (frame / fps) * 1000;
    const totalDuration = getTotalDuration();
    return Math.min(
      Math.floor((timeInMs / totalDuration) * images.length),
      images.length - 1
    );
  };

  const currentCaption = getCurrentCaption(frame);
  const currentImageIndex = getCurrentImageIndex(frame);

  const imageTransition = (index: number) => {
    const startTime = (index * getTotalDuration()) / images.length;
    const endTime = ((index + 1) * getTotalDuration()) / images.length;
    const progress = interpolate(
      frame,
      [(startTime / 1000) * fps, (endTime / 1000) * fps],
      [0, 1],
      {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp",
        easing: Easing.inOut(Easing.ease),
      }
    );

    return {
      opacity: interpolate(progress, [0, 0.1, 0.9, 1], [0, 1, 1, 0]),
      scale: interpolate(progress, [0, 1], [1.1, 1]),
    };
  };

  const fadeOutOpacity = interpolate(
    frame,
    [getDurationInFrames() - 30, getDurationInFrames()],
    [1, 0],
    {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    }
  );

  return (
    <AbsoluteFill style={{ backgroundColor: "#1a1a1a" }}>
      <Sequence from={0} durationInFrames={90}>
        <h1
          style={{
            fontSize: 60,
            fontFamily: fontBold,
            textAlign: "center",
            position: "absolute",
            top: "40%",
            width: "100%",
            color: "white",
          }}
        >
          {topic}
        </h1>
      </Sequence>

      {images.map((image, index) => {
        const { opacity, scale } = imageTransition(index);
        return (
          <Sequence
            key={index}
            from={0}
            durationInFrames={getDurationInFrames()}
          >
            <Img
              src={image}
              style={{
                width: "100%",
                height: "100%",
                objectFit: "cover",
                opacity: opacity * fadeOutOpacity,
                transform: `scale(${scale})`,
              }}
            />
          </Sequence>
        );
      })}

      {speechUrl && <Audio src={speechUrl} />}

      {currentCaption && (
        <div
          style={{
            position: "absolute",
            bottom: 50,
            left: 0,
            right: 0,
            textAlign: "center",
            backgroundColor: "rgba(0, 0, 0, 0.7)",
            padding: 20,
            opacity: fadeOutOpacity,
          }}
        >
          <p
            style={{
              color: "white",
              fontSize: 36,
              fontFamily: fontNormal,
              margin: 0,
              textShadow: "2px 2px 4px rgba(0,0,0,0.5)",
            }}
          >
            {currentCaption.text}
          </p>
        </div>
      )}
    </AbsoluteFill>
  );
};

export const Root: React.FC = () => {
  return (
    <Composition
      id="VideoGeneration"
      component={VideoComponent}
      durationInFrames={300}
      fps={30}
      width={1920}
      height={1080}
      defaultProps={defaultProps}
      calculateMetadata={async ({ props }) => {
        const audioDuration = await getAudioDurationInSeconds(props.speechUrl);
        const durationInSeconds = Math.max(
          props.captions[props.captions.length - 1].end / 1000,
          audioDuration,
          props.script.length * 5
        );
        return {
          durationInFrames: Math.ceil(durationInSeconds * 30),
          fps: 30,
        };
      }}
    />
  );
};
