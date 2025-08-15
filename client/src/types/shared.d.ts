// src/types/shared.d.ts
declare module "@shared/schema" {
  // src/shared/schema.ts

  export type UserResponses = {
    formality_level: number;
    friendliness_level: number;
    emotion_level: number;
    directness_level: number;
    uses_abbreviations: boolean;
    uses_emoticons: boolean;
    gratitude_expressions: string[];
    request_expressions: string[];
    situation_responses: Record<string, unknown>;
  };

  export type UserProfile = {
    id: number;
    userId: string;
    baseFormalityLevel: number;
    baseFriendlinessLevel: number;
    baseEmotionLevel: number;
    baseDirectnessLevel: number;
    responses: UserResponses;
    completedAt: Date;
  };

  export type ConversionRequest = {
    userId: string;
    inputText: string;
    targetStyleTags?: string[];
    options?: {
      preserveEmojis?: boolean;
      preserveAbbreviations?: boolean;
    };
  };
}
