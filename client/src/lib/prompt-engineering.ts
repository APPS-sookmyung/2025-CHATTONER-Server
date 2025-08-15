import type { UserProfile, ConversionRequest } from "@shared/schema";

export const NEGATIVE_PROMPTS = {
  chatgpt_style: [
    "물론입니다",
    "도움이 되셨기를 바랍니다",
    "추가로 궁금한 점이 있으시면",
    "기꺼이 도와드리겠습니다",
    "이해해 주셔서 감사합니다",
  ],
  overly_formal: [
    "귀하께서는",
    "말씀드리자면",
    "소중한 시간을 내어",
    "진심으로 감사드리며",
    "정중히 부탁드립니다",
  ],
  unnecessary_additions: [
    "참고로",
    "덧붙이자면",
    "한편으로는",
    "그런 측면에서",
    "종합해보면",
    "결론적으로 말씀드리면",
  ],
};

export function analyzeUserProfile(userProfile: UserProfile) {
  const responses = userProfile.responses as any;

  return {
    formalityDescription: getFormalityDescription(
      userProfile.baseFormalityLevel
    ),
    friendlinessDescription: getFriendlinessDescription(
      userProfile.baseFriendlinessLevel
    ),
    emotionDescription: getEmotionDescription(userProfile.baseEmotionLevel),
    usesAbbreviations:
      responses.abbreviation_usage &&
      ["자주 사용", "매우 자주 사용"].includes(responses.abbreviation_usage[0]),
    usesEmoticons:
      responses.emoticon_usage &&
      ["자주 사용", "매우 자주 사용", "문장마다 거의 사용"].includes(
        responses.emoticon_usage[0]
      ),
    preferredExpressions: {
      gratitude: responses.gratitude_senior || [],
      request: responses.request_colleague || [],
      closing: responses.closing_expressions || [],
      agreement: responses.agreement_expressions || [],
    },
  };
}

function getFormalityDescription(level: number): string {
  if (level >= 8) return "매우 정중한 격식체";
  if (level >= 6) return "정중한 편";
  if (level >= 4) return "보통 수준";
  if (level >= 2) return "캐주얼한 편";
  return "매우 캐주얼";
}

function getFriendlinessDescription(level: number): string {
  if (level >= 8) return "매우 친근하고 따뜻함";
  if (level >= 6) return "친근한 편";
  if (level >= 4) return "보통 수준";
  if (level >= 2) return "다소 딱딱한 편";
  return "매우 공식적";
}

function getEmotionDescription(level: number): string {
  if (level >= 8) return "감정 표현이 풍부함";
  if (level >= 6) return "적절한 감정 표현";
  if (level >= 4) return "보통 수준";
  if (level >= 2) return "담담한 편";
  return "매우 절제된 표현";
}

export function validateConversionInput(inputText: string): {
  isValid: boolean;
  error?: string;
} {
  if (!inputText.trim()) {
    return { isValid: false, error: "텍스트를 입력해주세요." };
  }

  if (inputText.length < 10) {
    return { isValid: false, error: "최소 10자 이상 입력해주세요." };
  }

  if (inputText.length > 2000) {
    return { isValid: false, error: "2000자 이하로 입력해주세요." };
  }

  return { isValid: true };
}

export function getContextDescription(context: string): string {
  switch (context) {
    case "general":
      return "일반적인 대화나 메시지";
    case "report":
      return "보고서, 업무 문서, 공식 문서";
    case "education":
      return "교육 자료, 설명서, 안내문";
    case "social":
      return "SNS 게시물, 소셜 미디어";
    default:
      return "일반";
  }
}
