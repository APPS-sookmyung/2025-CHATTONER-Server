import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Loader2, Sparkles, Trash2, RotateCcw } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/queryClient";
import OutputVersion from "./output-version";
import { validateConversionInput, getContextDescription } from "@/lib/prompt-engineering";
import type { UserProfile, ConversionRequest, ConversionResponse } from "@shared/schema";

interface TextConverterProps {
  userProfile: UserProfile;
  userId: string;
}

export default function TextConverter({ userProfile, userId }: TextConverterProps) {
  const { toast } = useToast();
  const [inputText, setInputText] = useState('');
  const [selectedContext, setSelectedContext] = useState<'general' | 'report' | 'education' | 'social'>('general');
  const [conversionOptions, setConversionOptions] = useState({
    removeChatgptStyle: true,
    addEmoticons: false,
    useAbbreviations: false,
  });
  const [conversionResult, setConversionResult] = useState<ConversionResponse | null>(null);
  const [currentHistoryId, setCurrentHistoryId] = useState<number | null>(null);

  // Fetch conversion history
  const { data: history } = useQuery({
    queryKey: [`/api/history/${userId}`],
    enabled: !!userId,
  });

  // Text conversion mutation
  const convertMutation = useMutation({
    mutationFn: async (data: ConversionRequest & { userId: string }) => {
      const response = await apiRequest('POST', '/api/convert', data);
      return response.json();
    },
    onSuccess: (data) => {
      setConversionResult({
        standard: data.standard,
        casual: data.casual,
        formal: data.formal,
      });
      setCurrentHistoryId(data.historyId);
      toast({
        title: "변환 완료",
        description: "텍스트가 성공적으로 변환되었습니다.",
      });
    },
    onError: (error: any) => {
      toast({
        variant: "destructive",
        title: "변환 실패",
        description: error.message || "텍스트 변환 중 오류가 발생했습니다.",
      });
    },
  });

  // Feedback mutation
  const feedbackMutation = useMutation({
    mutationFn: async ({ historyId, version, type }: { historyId: number; version: string; type: string }) => {
      const response = await apiRequest('POST', `/api/feedback/${historyId}`, { version, type });
      return response.json();
    },
    onSuccess: () => {
      toast({
        title: "피드백 저장됨",
        description: "소중한 피드백이 저장되었습니다.",
      });
    },
  });

  const handleConvert = () => {
    const validation = validateConversionInput(inputText);
    if (!validation.isValid) {
      toast({
        variant: "destructive",
        title: "입력 오류",
        description: validation.error,
      });
      return;
    }

    const requestData: ConversionRequest & { userId: string } = {
      userId,
      inputText,
      context: selectedContext,
      options: conversionOptions,
    };

    convertMutation.mutate(requestData);
  };

  const handleClear = () => {
    setInputText('');
    setConversionResult(null);
    setCurrentHistoryId(null);
  };

  const handleFeedback = (version: string, type: 'like' | 'dislike') => {
    if (currentHistoryId) {
      feedbackMutation.mutate({ historyId: currentHistoryId, version, type });
    }
  };

  const handleCopy = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      toast({
        title: "복사 완료",
        description: "텍스트가 클립보드에 복사되었습니다.",
      });
    } catch (error) {
      toast({
        variant: "destructive",
        title: "복사 실패",
        description: "클립보드 복사에 실패했습니다.",
      });
    }
  };

  const characterCount = inputText.length;

  return (
    <div className="grid lg:grid-cols-2 gap-8">
      {/* Input Section */}
      <div className="space-y-6">
        <Card className="shadow-sm border border-gray-200">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">텍스트 입력</h3>
              <div className="flex items-center space-x-2">
                <Sparkles className="text-primary w-4 h-4" />
                <span className="text-sm text-gray-600">AI 변환</span>
              </div>
            </div>
            
            {/* Context Selector */}
            <div className="mb-4">
              <Label className="text-sm font-medium text-gray-700 mb-2 block">상황 선택</Label>
              <Select value={selectedContext} onValueChange={setSelectedContext}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="general">일반 대화</SelectItem>
                  <SelectItem value="report">보고서/업무</SelectItem>
                  <SelectItem value="education">교육 콘텐츠</SelectItem>
                  <SelectItem value="social">소셜 미디어</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-gray-500 mt-1">
                {getContextDescription(selectedContext)}
              </p>
            </div>
            
            {/* Text Input */}
            <div className="mb-4">
              <Textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="변환할 텍스트를 입력하세요...

예시: 이번 프로젝트의 결과를 보고드리겠습니다. 상당히 만족스러운 성과를 거두었다고 판단됩니다."
                className="h-40 resize-none"
                maxLength={2000}
              />
            </div>
            
            {/* Character Count & Actions */}
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-500">{characterCount}자</span>
              <div className="flex space-x-3">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleClear}
                  className="text-gray-600 hover:text-gray-800"
                >
                  <Trash2 className="w-4 h-4 mr-2" />
                  지우기
                </Button>
                <Button
                  onClick={handleConvert}
                  disabled={!inputText.trim() || convertMutation.isPending}
                  className="px-6"
                >
                  {convertMutation.isPending ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      변환 중...
                    </>
                  ) : (
                    <>
                      <RotateCcw className="w-4 h-4 mr-2" />
                      변환하기
                    </>
                  )}
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Conversion Settings */}
        <Card className="border border-gray-200">
          <CardContent className="p-6">
            <h4 className="font-semibold text-gray-900 mb-4">변환 옵션</h4>
            
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <Label htmlFor="remove-chatgpt" className="text-sm text-gray-700">
                  ChatGPT 스타일 제거
                </Label>
                <Switch
                  id="remove-chatgpt"
                  checked={conversionOptions.removeChatgptStyle}
                  onCheckedChange={(checked) => 
                    setConversionOptions(prev => ({ ...prev, removeChatgptStyle: checked }))
                  }
                />
              </div>
              
              <div className="flex items-center justify-between">
                <Label htmlFor="add-emoticons" className="text-sm text-gray-700">
                  이모티콘 추가
                </Label>
                <Switch
                  id="add-emoticons"
                  checked={conversionOptions.addEmoticons}
                  onCheckedChange={(checked) => 
                    setConversionOptions(prev => ({ ...prev, addEmoticons: checked }))
                  }
                />
              </div>
              
              <div className="flex items-center justify-between">
                <Label htmlFor="use-abbreviations" className="text-sm text-gray-700">
                  줄임말 사용
                </Label>
                <Switch
                  id="use-abbreviations"
                  checked={conversionOptions.useAbbreviations}
                  onCheckedChange={(checked) => 
                    setConversionOptions(prev => ({ ...prev, useAbbreviations: checked }))
                  }
                />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Output Section */}
      <div className="space-y-6">
        <Card className="shadow-sm border border-gray-200">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-gray-900">변환 결과</h3>
              {conversionResult && (
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-success rounded-full" />
                  <span className="text-sm text-success font-medium">변환 완료</span>
                </div>
              )}
            </div>

            {conversionResult ? (
              <div className="space-y-4">
                <OutputVersion
                  type="standard"
                  label="표준"
                  description="기본 적용"
                  text={conversionResult.standard}
                  onLike={() => handleFeedback('standard', 'like')}
                  onDislike={() => handleFeedback('standard', 'dislike')}
                  onCopy={() => handleCopy(conversionResult.standard)}
                />
                <OutputVersion
                  type="casual"
                  label="캐주얼"
                  description="더 편한 톤"
                  text={conversionResult.casual}
                  onLike={() => handleFeedback('casual', 'like')}
                  onDislike={() => handleFeedback('casual', 'dislike')}
                  onCopy={() => handleCopy(conversionResult.casual)}
                />
                <OutputVersion
                  type="formal"
                  label="정중"
                  description="더 격식있는 톤"
                  text={conversionResult.formal}
                  onLike={() => handleFeedback('formal', 'like')}
                  onDislike={() => handleFeedback('formal', 'dislike')}
                  onCopy={() => handleCopy(conversionResult.formal)}
                />
              </div>
            ) : (
              <div className="text-center py-12 text-gray-500">
                <Sparkles className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                <p>텍스트를 입력하고 변환 버튼을 눌러주세요</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Conversion History */}
        {history && history.length > 0 && (
          <Card className="border border-gray-200">
            <CardContent className="p-6">
              <h4 className="font-semibold text-gray-900 mb-4">최근 변환 기록</h4>
              
              <div className="space-y-3">
                {history.slice(0, 3).map((item: any) => (
                  <div key={item.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div className="flex-1">
                      <p className="text-sm text-gray-800 truncate">
                        {item.inputText.slice(0, 50)}...
                      </p>
                      <p className="text-xs text-gray-500 mt-1">
                        {new Date(item.createdAt).toLocaleString()} • {getContextDescription(item.context)}
                      </p>
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setInputText(item.inputText)}
                    >
                      <RotateCcw className="w-4 h-4" />
                    </Button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
