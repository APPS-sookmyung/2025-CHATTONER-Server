import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Wand2, CheckCircle, ArrowRight } from 'lucide-react';

interface ModeSelectorProps {
  onModeSelect: (mode: 'transform' | 'validate') => void;
}

export default function ModeSelector({ onModeSelect }: ModeSelectorProps) {
  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div className="text-center space-y-4">
        <h2 className="text-3xl font-bold text-gray-900 dark:text-white">
          어떤 기능을 사용하시겠습니까?
        </h2>
        <p className="text-lg text-gray-600 dark:text-gray-300">
          목적에 맞는 도구를 선택해주세요
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Style Transform Mode */}
        <Card className="relative hover:shadow-lg transition-all duration-200 cursor-pointer group"
              onClick={() => onModeSelect('transform')}>
          <CardHeader className="pb-4">
            <div className="flex items-center space-x-3">
              <div className="p-3 bg-blue-100 dark:bg-blue-900 rounded-lg">
                <Wand2 className="w-6 h-6 text-blue-600 dark:text-blue-400" />
              </div>
              <div>
                <CardTitle className="text-xl">🎭 스타일 변환</CardTitle>
                <CardDescription className="text-sm text-gray-500">
                  말투나 톤을 바꾸고 싶을 때
                </CardDescription>
              </div>
            </div>
          </CardHeader>
          
          <CardContent className="space-y-4">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              이미 작성된 텍스트의 말투나 톤을 상황에 맞게 변환합니다.
            </p>
            
            <div className="space-y-2">
              <h4 className="font-medium text-sm">이런 경우에 사용하세요:</h4>
              <ul className="text-xs text-gray-500 space-y-1">
                <li>• 캐주얼한 메시지를 정중하게 바꾸고 싶을 때</li>
                <li>• 딱딱한 문서를 친근하게 만들고 싶을 때</li>
                <li>• 상황에 맞는 말투로 조정하고 싶을 때</li>
              </ul>
            </div>

            <Button 
              className="w-full group-hover:bg-blue-600 transition-colors"
              onClick={(e) => {
                e.stopPropagation();
                onModeSelect('transform');
              }}
            >
              스타일 변환하기
              <ArrowRight className="w-4 h-4 ml-2" />
            </Button>
          </CardContent>
        </Card>

        {/* Quality Validation Mode */}
        <Card className="relative hover:shadow-lg transition-all duration-200 cursor-pointer group"
              onClick={() => onModeSelect('validate')}>
          <CardHeader className="pb-4">
            <div className="flex items-center space-x-3">
              <div className="p-3 bg-green-100 dark:bg-green-900 rounded-lg">
                <CheckCircle className="w-6 h-6 text-green-600 dark:text-green-400" />
              </div>
              <div>
                <CardTitle className="text-xl">✅ 품질 검증</CardTitle>
                <CardDescription className="text-sm text-gray-500">
                  완성된 글이 적절한지 확인하고 싶을 때
                </CardDescription>
              </div>
            </div>
          </CardHeader>
          
          <CardContent className="space-y-4">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              완성된 텍스트가 맥락에 적절한지 분석하고 개선점을 제안합니다.
            </p>
            
            <div className="space-y-2">
              <h4 className="font-medium text-sm">이런 경우에 사용하세요:</h4>
              <ul className="text-xs text-gray-500 space-y-1">
                <li>• 보고서나 공문의 적절성을 확인하고 싶을 때</li>
                <li>• 맞춤법이나 문법을 점검하고 싶을 때</li>
                <li>• 더 나은 표현으로 개선하고 싶을 때</li>
              </ul>
            </div>

            <Button 
              className="w-full bg-green-600 hover:bg-green-700 group-hover:bg-green-700 transition-colors"
              onClick={(e) => {
                e.stopPropagation();
                onModeSelect('validate');
              }}
            >
              품질 검증하기
              <ArrowRight className="w-4 h-4 ml-2" />
            </Button>
          </CardContent>
        </Card>
      </div>

      <div className="bg-blue-50 dark:bg-blue-950 p-6 rounded-lg">
        <div className="flex items-start space-x-3">
          <div className="text-blue-600 dark:text-blue-400 mt-1">💡</div>
          <div>
            <h3 className="font-medium text-blue-900 dark:text-blue-100 mb-2">
              어떤 도구를 선택해야 할까요?
            </h3>
            <div className="text-sm text-blue-800 dark:text-blue-200 space-y-1">
              <p><strong>스타일 변환:</strong> 내용은 좋은데 말투만 바꾸고 싶을 때</p>
              <p><strong>품질 검증:</strong> 내용 자체를 점검하고 개선하고 싶을 때</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}