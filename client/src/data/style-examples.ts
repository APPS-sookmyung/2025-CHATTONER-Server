export interface StyleExample {
  id: string;
  situation: string;
  options: {
    A: {
      text: string;
      style: 'direct' | 'gentle' | 'neutral';
      characteristics: string[];
    };
    B: {
      text: string;
      style: 'direct' | 'gentle' | 'neutral';
      characteristics: string[];
    };
    C: {
      text: string;
      style: 'direct' | 'gentle' | 'neutral';
      characteristics: string[];
    };
  };
}

export const styleExamples: StyleExample[] = [
  {
    id: "meeting_late",
    situation: "회의에 늦었을 때 사과하는 상황",
    options: {
      A: {
        text: "늦어서 죄송합니다.",
        style: 'direct',
        characteristics: ['간결함', '직설적', '공식적']
      },
      B: {
        text: "정말 죄송해요! 교통이 너무 막혀서 늦었어요. 많이 기다리셨죠?",
        style: 'gentle',
        characteristics: ['상세한 설명', '감정적', '친근함']
      },
      C: {
        text: "죄송합니다. 교통 체증으로 늦었습니다.",
        style: 'neutral',
        characteristics: ['적당한 설명', '중립적', '정중함']
      }
    }
  },
  {
    id: "request_help",
    situation: "동료에게 도움을 요청하는 상황",
    options: {
      A: {
        text: "이거 좀 도와주세요.",
        style: 'direct',
        characteristics: ['간결함', '직접적', '간단명료']
      },
      B: {
        text: "혹시 시간 괜찮으시면 이 부분 좀 도와주실 수 있을까요? 제가 잘 모르는 부분이라서요.",
        style: 'gentle',
        characteristics: ['정중함', '배려심', '겸손함']
      },
      C: {
        text: "이 부분에 대해 도움이 필요합니다. 시간 되실 때 도와주시면 감사하겠습니다.",
        style: 'neutral',
        characteristics: ['공손함', '명확함', '전문적']
      }
    }
  },
  {
    id: "disagree_opinion",
    situation: "의견이 다를 때 반대 의사를 표현하는 상황",
    options: {
      A: {
        text: "그건 아닌 것 같은데요.",
        style: 'direct',
        characteristics: ['솔직함', '직설적', '간결함']
      },
      B: {
        text: "말씀하신 부분도 좋은데, 제가 생각하기에는 조금 다른 방법도 있을 것 같아요. 어떻게 생각하세요?",
        style: 'gentle',
        characteristics: ['완곡함', '배려심', '대화형']
      },
      C: {
        text: "좋은 의견이지만, 다른 관점에서 보면 이런 방법도 고려해볼 수 있을 것 같습니다.",
        style: 'neutral',
        characteristics: ['객관적', '균형적', '논리적']
      }
    }
  },
  {
    id: "express_gratitude",
    situation: "상사에게 감사 인사를 전하는 상황",
    options: {
      A: {
        text: "감사합니다.",
        style: 'direct',
        characteristics: ['간결함', '공식적', '명확함']
      },
      B: {
        text: "정말 많이 도와주셔서 감사해요! 덕분에 잘 해결할 수 있었어요.",
        style: 'gentle',
        characteristics: ['감정적', '구체적', '따뜻함']
      },
      C: {
        text: "도움 주셔서 감사합니다. 덕분에 일이 순조롭게 진행되었습니다.",
        style: 'neutral',
        characteristics: ['정중함', '구체적', '전문적']
      }
    }
  },
  {
    id: "give_feedback",
    situation: "부하직원에게 피드백을 주는 상황",
    options: {
      A: {
        text: "이 부분은 수정이 필요합니다.",
        style: 'direct',
        characteristics: ['명확함', '직접적', '간결함']
      },
      B: {
        text: "전체적으로 잘 하셨는데, 이 부분만 조금 더 보완하면 더 좋을 것 같아요. 어떻게 생각하세요?",
        style: 'gentle',
        characteristics: ['격려적', '부드러움', '협력적']
      },
      C: {
        text: "좋은 작업이었습니다. 다만 이 부분에서 개선이 필요해 보입니다.",
        style: 'neutral',
        characteristics: ['균형적', '건설적', '객관적']
      }
    }
  }
];

export function analyzeStylePreferences(choices: Record<string, 'A' | 'B' | 'C'>): {
  directness: number;
  formality: number;
  emotion: number;
  friendliness: number;
} {
  let directCount = 0;
  let gentleCount = 0;
  let neutralCount = 0;
  
  Object.values(choices).forEach(choice => {
    const example = styleExamples.find(ex => ex.id in choices);
    if (example) {
      const selectedOption = example.options[choice];
      switch (selectedOption.style) {
        case 'direct':
          directCount++;
          break;
        case 'gentle':
          gentleCount++;
          break;
        case 'neutral':
          neutralCount++;
          break;
      }
    }
  });
  
  const total = directCount + gentleCount + neutralCount;
  
  return {
    directness: total > 0 ? Math.round((directCount / total) * 10) : 5,
    formality: total > 0 ? Math.round(((directCount + neutralCount) / total) * 10) : 5,
    emotion: total > 0 ? Math.round((gentleCount / total) * 10) : 5,
    friendliness: total > 0 ? Math.round((gentleCount / total) * 10) : 5,
  };
}