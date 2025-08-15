import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { ArrowLeft, BarChart3, CheckCircle, AlertTriangle, Lightbulb, Copy } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { apiRequest } from '@/lib/queryClient';

interface QualityValidatorProps {
  onBack: () => void;
}

interface QualityAnalysis {
  grammar_score: number;
  formality_score: number;
  readability_score: number;
  suggestions: Array<{
    type: string;
    original: string;
    suggestion: string;
    reason: string;
    confidence: number;
  }>;
}

export default function QualityValidator({ onBack }: QualityValidatorProps) {
  const [inputText, setInputText] = useState('');
  const [context, setContext] = useState<'business' | 'report' | 'casual'>('business');
  const [analysis, setAnalysis] = useState<QualityAnalysis | null>(null);
  const { toast } = useToast();

  const analyzeMutation = useMutation({
    mutationFn: async (text: string): Promise<QualityAnalysis> => {
      const response = await apiRequest('/api/analyze-quality', {
        method: 'POST',
        body: JSON.stringify({ text }),
      });
      return response.json();
    },
    onSuccess: (data) => {
      setAnalysis(data);
    },
    onError: (error) => {
      toast({
        title: '분석 실패',
        description: error instanceof Error ? error.message : '알 수 없는 오류가 발생했습니다.',
        variant: 'destructive',
      });
    },
  });

  const suggestionsMutation = useMutation({
    mutationFn: async ({ text, context }: { text: string; context: string }) => {
      const response = await apiRequest('/api/context-suggestions', {
        method: 'POST',
        body: JSON.stringify({ text, context }),
      });
      return response.json();
    },
  });

  const handleAnalyze = () => {
    if (!inputText.trim()) {
      toast({
        title: '텍스트를 입력해주세요',
        description: '분석할 텍스트를 입력해주세요.',
        variant: 'destructive',
      });
      return;
    }

    analyzeMutation.mutate(inputText.trim());
    suggestionsMutation.mutate({ text: inputText.trim(), context });
  };

  const applySuggestion = (original: string, suggestion: string) => {
    const newText = inputText.replace(original, suggestion);
    setInputText(newText);
    toast({
      title: '개선 적용',
      description: `"${original}" → "${suggestion}"으로 변경되었습니다.`,
    });
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast({
      title: '복사 완료',
      description: '텍스트가 클립보드에 복사되었습니다.',
    });
  };

  const getScoreColor = (score: number) => {
    if (score >= 8) return 'text-green-600 dark:text-green-400';
    if (score >= 6) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  const getScoreBadge = (score: number) => {
    if (score >= 8) return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
    if (score >= 6) return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
    return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center space-x-4">
        <Button variant="outline" onClick={onBack}>
          <ArrowLeft className="w-4 h-4 mr-2" />
          뒤로가기
        </Button>
        <div>
          <h1 className="text-2xl font-bold">✅ 품질 검증</h1>
          <p className="text-gray-600 dark:text-gray-400">
            텍스트의 품질을 분석하고 개선점을 제안합니다
          </p>
        </div>
      </div>

      {/* Input Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="w-5 h-5" />
            텍스트 분석
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Textarea
            placeholder="분석할 텍스트를 입력하세요..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            className="min-h-[150px]"
          />
          
          <div className="flex gap-4 items-end">
            <div className="flex-1">
              <label className="text-sm font-medium mb-2 block">분석 맥락</label>
              <Select value={context} onValueChange={(value: 'business' | 'report' | 'casual') => setContext(value)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="business">업무/비즈니스</SelectItem>
                  <SelectItem value="report">보고서/공문</SelectItem>
                  <SelectItem value="casual">일상/캐주얼</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <Button 
              onClick={handleAnalyze}
              disabled={analyzeMutation.isPending}
              className="min-w-[120px]"
            >
              {analyzeMutation.isPending ? '분석 중...' : '품질 분석'}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Analysis Results */}
      {analysis && (
        <div className="grid md:grid-cols-2 gap-6">
          {/* Scores */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="w-5 h-5" />
                분석 점수
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-3 gap-4">
                <div className="text-center">
                  <div className={`text-2xl font-bold ${getScoreColor(analysis.grammar_score)}`}>
                    {analysis.grammar_score}/10
                  </div>
                  <Badge variant="outline" className={getScoreBadge(analysis.grammar_score)}>
                    문법
                  </Badge>
                </div>
                <div className="text-center">
                  <div className={`text-2xl font-bold ${getScoreColor(analysis.formality_score)}`}>
                    {analysis.formality_score}/10
                  </div>
                  <Badge variant="outline" className={getScoreBadge(analysis.formality_score)}>
                    격식도
                  </Badge>
                </div>
                <div className="text-center">
                  <div className={`text-2xl font-bold ${getScoreColor(analysis.readability_score)}`}>
                    {analysis.readability_score}/10
                  </div>
                  <Badge variant="outline" className={getScoreBadge(analysis.readability_score)}>
                    가독성
                  </Badge>
                </div>
              </div>

              <div className="pt-4 border-t">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 dark:text-gray-400">개선된 텍스트</span>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => copyToClipboard(inputText)}
                  >
                    <Copy className="w-4 h-4 mr-2" />
                    복사
                  </Button>
                </div>
                <div className="mt-2 p-3 bg-gray-50 dark:bg-gray-800 rounded-lg text-sm">
                  {inputText}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Suggestions */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Lightbulb className="w-5 h-5" />
                개선 제안
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {analysis.suggestions.length > 0 ? (
                analysis.suggestions.map((suggestion, index) => (
                  <div key={index} className="p-3 border rounded-lg space-y-2">
                    <div className="flex items-center justify-between">
                      <Badge variant="outline" className="text-xs">
                        {suggestion.type === 'grammar' ? '문법' : 
                         suggestion.type === 'business' ? '업무용어' : '가독성'}
                      </Badge>
                      <span className="text-xs text-gray-500">
                        신뢰도: {Math.round(suggestion.confidence * 100)}%
                      </span>
                    </div>
                    
                    <div className="text-sm">
                      <div className="flex items-center gap-2 text-red-600 dark:text-red-400">
                        <AlertTriangle className="w-3 h-3" />
                        <span className="font-mono bg-red-50 dark:bg-red-950 px-1 rounded">
                          {suggestion.original}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 text-green-600 dark:text-green-400 mt-1">
                        <CheckCircle className="w-3 h-3" />
                        <span className="font-mono bg-green-50 dark:bg-green-950 px-1 rounded">
                          {suggestion.suggestion}
                        </span>
                      </div>
                    </div>
                    
                    <div className="text-xs text-gray-600 dark:text-gray-400">
                      {suggestion.reason}
                    </div>
                    
                    <Button
                      size="sm"
                      variant="outline"
                      className="w-full"
                      onClick={() => applySuggestion(suggestion.original, suggestion.suggestion)}
                    >
                      적용하기
                    </Button>
                  </div>
                ))
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <CheckCircle className="w-12 h-12 mx-auto mb-2 text-green-500" />
                  <p>개선할 점이 발견되지 않았습니다!</p>
                  <p className="text-sm">텍스트 품질이 우수합니다.</p>
                </div>
              )}

              {suggestionsMutation.data?.suggestions?.length > 0 && (
                <div className="pt-4 border-t">
                  <h4 className="font-medium text-sm mb-2">맥락별 추가 제안</h4>
                  {suggestionsMutation.data.suggestions.map((suggestion: any, index: number) => (
                    <div key={index} className="p-2 bg-blue-50 dark:bg-blue-950 rounded text-xs">
                      <div className="font-mono">
                        "{suggestion.original}" → "{suggestion.suggestion}"
                      </div>
                      <div className="text-gray-600 dark:text-gray-400 mt-1">
                        {suggestion.reason}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}