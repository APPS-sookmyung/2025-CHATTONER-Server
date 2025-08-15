export interface Question {
  id: string;
  category: 'situational' | 'preferences' | 'expressions';
  categoryLabel: string;
  question: string;
  description: string;
  options: string[];
  allowMultiple: boolean;
  allowCustom: boolean;
}

export const questions: Question[] = [
  // Situational Speaking Style
  {
    id: 'gratitude_senior',
    category: 'situational',
    categoryLabel: '상황별 말투',
    question: '선배님께 감사인사를 드릴 때 어떻게 표현하시나요?',
    description: '평소 자주 사용하는 표현을 모두 선택해주세요 (복수 선택 가능)',
    options: [
      '정말 감사합니다',
      '감사해요~',
      '고마워요',
      '진짜 고맙습니다',
      '너무 감사드려요'
    ],
    allowMultiple: true,
    allowCustom: true
  },
  {
    id: 'request_colleague',
    category: 'situational',
    categoryLabel: '상황별 말투',
    question: '동료에게 도움을 요청할 때는?',
    description: '업무나 일상에서 동료에게 부탁할 때 주로 사용하는 표현',
    options: [
      '혹시 시간 되시면 도와주실 수 있나요?',
      '도움 좀 받을 수 있을까요?',
      '이거 좀 도와줄래?',
      '바쁘시겠지만 잠깐 도와주시면 감사하겠습니다',
      '부탁 하나 있는데 괜찮으세요?'
    ],
    allowMultiple: true,
    allowCustom: true
  },
  {
    id: 'report_writing',
    category: 'situational',
    categoryLabel: '상황별 말투',
    question: '보고서나 공식 문서를 작성할 때?',
    description: '업무용 문서에서 선호하는 문체 스타일',
    options: [
      '~입니다/~습니다 (격식체)',
      '~해요/~예요 (반격식체)',
      '~함/~음 (간결체)',
      '~드립니다/~겠습니다 (높임체)',
      '~였습니다/~했습니다 (과거 존댓말)'
    ],
    allowMultiple: true,
    allowCustom: false
  },
  {
    id: 'friend_conversation',
    category: 'situational',
    categoryLabel: '상황별 말투',
    question: '친구들과 대화할 때는?',
    description: '편한 사이에서 주로 사용하는 말투',
    options: [
      '반말 (해/야)',
      '존댓말 (해요/이에요)',
      '상황에 따라 섞어서',
      '친근한 존댓말 (해요~ 식)',
      '방언이나 사투리'
    ],
    allowMultiple: true,
    allowCustom: true
  },
  {
    id: 'presentation_style',
    category: 'situational',
    categoryLabel: '상황별 말투',
    question: '공식적인 발표를 할 때?',
    description: '회의나 프레젠테이션에서의 선호 스타일',
    options: [
      '~습니다/~입니다 (표준 격식체)',
      '~해요/~예요 (부드러운 격식체)',
      '~겠습니다/~드리겠습니다 (정중 격식체)',
      '간결하고 직접적인 표현',
      '친근하면서도 격식있는 표현'
    ],
    allowMultiple: false,
    allowCustom: false
  },

  // Language Preferences (Scale-based)
  {
    id: 'formality_level',
    category: 'preferences',
    categoryLabel: '언어 선호도',
    question: '평소 말투의 정중함 정도는?',
    description: '1-10 척도로 선택해주세요 (1: 매우 캐주얼, 10: 매우 정중)',
    options: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    allowMultiple: false,
    allowCustom: false
  },
  {
    id: 'friendliness_level',
    category: 'preferences',
    categoryLabel: '언어 선호도',
    question: '친근함 정도는 어느 정도인가요?',
    description: '1-10 척도 (1: 매우 딱딱함, 10: 매우 친근함)',
    options: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    allowMultiple: false,
    allowCustom: false
  },
  {
    id: 'emotion_expression',
    category: 'preferences',
    categoryLabel: '언어 선호도',
    question: '감정 표현을 얼마나 자주 하시나요?',
    description: '1-10 척도 (1: 매우 담담함, 10: 감정 풍부)',
    options: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    allowMultiple: false,
    allowCustom: false
  },

  // Specific Expressions
  {
    id: 'abbreviation_usage',
    category: 'expressions',
    categoryLabel: '구체적 표현',
    question: '줄임말이나 인터넷 용어를 얼마나 사용하시나요?',
    description: '텍스트나 메신저에서의 사용 패턴',
    options: [
      '거의 사용하지 않음',
      '가끔 사용',
      '자주 사용',
      '매우 자주 사용',
      '상황에 따라 다름'
    ],
    allowMultiple: false,
    allowCustom: false
  },
  {
    id: 'emoticon_usage',
    category: 'expressions',
    categoryLabel: '구체적 표현',
    question: '이모티콘이나 이모지를 얼마나 사용하시나요?',
    description: '😊, ㅎㅎ, ^^ 등의 감정 표현',
    options: [
      '전혀 사용하지 않음',
      '가끔 사용 (중요할 때만)',
      '자주 사용',
      '매우 자주 사용',
      '문장마다 거의 사용'
    ],
    allowMultiple: false,
    allowCustom: false
  },
  {
    id: 'closing_expressions',
    category: 'expressions',
    categoryLabel: '구체적 표현',
    question: '대화를 마무리할 때 자주 사용하는 표현은?',
    description: '메시지나 대화의 끝맺음 표현',
    options: [
      '감사합니다',
      '고생하세요',
      '수고하세요',
      '그럼 이만~',
      '잘 부탁드려요',
      '연락드릴게요'
    ],
    allowMultiple: true,
    allowCustom: true
  },
  {
    id: 'agreement_expressions',
    category: 'expressions',
    categoryLabel: '구체적 표현',
    question: '동의하거나 긍정할 때 주로 사용하는 표현은?',
    description: '상대방 의견에 동의할 때의 반응',
    options: [
      '네, 맞습니다',
      '그렇네요',
      '맞아요',
      '인정!',
      '완전 동감',
      '좋은 생각이에요'
    ],
    allowMultiple: true,
    allowCustom: true
  }
];

export const getQuestionsByCategory = (category: Question['category']) => {
  return questions.filter(q => q.category === category);
};

export const getCategoryProgress = (category: Question['category'], responses: Record<string, any>) => {
  const categoryQuestions = getQuestionsByCategory(category);
  const answeredQuestions = categoryQuestions.filter(q => responses[q.id]);
  return {
    answered: answeredQuestions.length,
    total: categoryQuestions.length,
    percentage: Math.round((answeredQuestions.length / categoryQuestions.length) * 100)
  };
};
