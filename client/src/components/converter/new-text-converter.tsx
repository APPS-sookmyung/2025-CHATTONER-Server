import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Loader2, Wand2 } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { apiRequest } from '@/lib/queryClient';
import type { UserProfile, ConversionRequest, ConversionResponse } from '@shared/schema';
import NewOutputVersion from './new-output-version';

interface NewTextConverterProps {
  userProfile: UserProfile;
  userId: string;
}

export default function NewTextConverter({ userProfile, userId }: NewTextConverterProps) {
  const [inputText, setInputText] = useState('');
  const [context, setContext] = useState<'general' | 'report' | 'education' | 'social'>('general');
  const [lastConversionId, setLastConversionId] = useState<number | null>(null);
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const convertMutation = useMutation({
    mutationFn: async (request: ConversionRequest): Promise<ConversionResponse> => {
      const response = await apiRequest('/api/convert', {
        method: 'POST',
        body: JSON.stringify({ ...request, userId }),
      });
      
      const data = await response.json();
      if (data.conversionId) {
        setLastConversionId(data.conversionId);
      }
      return data;
    },
    onError: (error) => {
      toast({
        title: '변환 실패',
        description: error instanceof Error ? error.message : '알 수 없는 오류가 발생했습니다.',
        variant: 'destructive',
      });
    },
  });

  const feedbackMutation = useMutation({
    mutationFn: async ({ selectedVersion, userFeedback }: { 
      selectedVersion: 'direct' | 'gentle' | 'neutral';
      userFeedback?: string;
    }) => {
      const response = await apiRequest('/api/feedback', {
        method: 'POST',
        body: JSON.stringify({
          conversionId: lastConversionId,
          selectedVersion,
          userFeedback,
          userId,
        }),
      });
      

      
      return response.json();
    },
    onSuccess: () => {
      // Refresh user profile to get updated session style
      queryClient.invalidateQueries({ queryKey: ['/api/profile', userId] });
    },
  });

  const handleConvert = () => {
    if (!inputText.trim()) {
      toast({
        title: '텍스트를 입력해주세요',
        description: '변환할 텍스트를 입력해주세요.',
        variant: 'destructive',
      });
      return;
    }

    convertMutation.mutate({
      inputText: inputText.trim(),
      context,
      options: {
        removeChatgptStyle: true,
        addEmoticons: false,
        useAbbreviations: false,
      },
    });
  };

  const handleVersionSelect = (version: 'direct' | 'gentle' | 'neutral', feedback?: string) => {
    feedbackMutation.mutate({
      selectedVersion: version,
      userFeedback: feedback,
    });
  };

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    toast({
      title: '복사 완료',
      description: '텍스트가 클립보드에 복사되었습니다.',
    });
  };

  const getContextLabel = (ctx: string) => {
    const labels = {
      general: '일반',
      report: '보고서/문서',
      education: '교육/설명',
      social: '소셜미디어'
    };
    return labels[ctx as keyof typeof labels] || ctx;
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Wand2 className="w-5 h-5" />
            개성 맞춤 텍스트 변환기
          </CardTitle>
          <div className="flex gap-2">
            <Badge variant="outline">
              격식도: {userProfile.sessionFormalityLevel || userProfile.baseFormalityLevel}/10
            </Badge>
            <Badge variant="outline">
              친근함: {userProfile.sessionFriendlinessLevel || userProfile.baseFriendlinessLevel}/10
            </Badge>
            <Badge variant="outline">
              감정표현: {userProfile.sessionEmotionLevel || userProfile.baseEmotionLevel}/10
            </Badge>
            <Badge variant="outline">
              직설성: {userProfile.sessionDirectnessLevel || userProfile.baseDirectnessLevel}/10
            </Badge>
          </div>
        </CardHeader>
        
        <CardContent className="space-y-4">
          <div>
            <label className="text-sm font-medium">변환할 텍스트</label>
            <Textarea
              placeholder="변환하고 싶은 텍스트를 입력하세요..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              rows={4}
              className="mt-1"
            />
          </div>
          
          <div>
            <label className="text-sm font-medium">상황/맥락</label>
            <Select value={context} onValueChange={(value: any) => setContext(value)}>
              <SelectTrigger className="mt-1">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="general">일반 대화</SelectItem>
                <SelectItem value="report">보고서/문서</SelectItem>
                <SelectItem value="education">교육/설명</SelectItem>
                <SelectItem value="social">소셜미디어</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          <Button
            onClick={handleConvert}
            disabled={convertMutation.isPending || !inputText.trim()}
            className="w-full"
          >
            {convertMutation.isPending ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                변환 중...
              </>
            ) : (
              <>
                <Wand2 className="w-4 h-4 mr-2" />
                3가지 스타일로 변환하기
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {convertMutation.data && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">변환 결과 - {getContextLabel(context)}</h3>
          <div className="grid gap-6 md:grid-cols-3">
            <NewOutputVersion
              type="direct"
              label="직설적 버전"
              description="간결하고 명확하며 직접적인 표현"
              text={convertMutation.data.direct}
              conversionId={lastConversionId || undefined}
              onSelect={handleVersionSelect}
              onCopy={() => handleCopy(convertMutation.data!.direct)}
            />
            
            <NewOutputVersion
              type="gentle"
              label="부드러운 버전"
              description="완곡하고 배려심 있으며 친근한 표현"
              text={convertMutation.data.gentle}
              conversionId={lastConversionId || undefined}
              onSelect={handleVersionSelect}
              onCopy={() => handleCopy(convertMutation.data!.gentle)}
            />
            
            <NewOutputVersion
              type="neutral"
              label="중립적 버전"
              description="균형 잡히고 객관적이며 적당히 정중한 표현"
              text={convertMutation.data.neutral}
              conversionId={lastConversionId || undefined}
              onSelect={handleVersionSelect}
              onCopy={() => handleCopy(convertMutation.data!.neutral)}
            />
          </div>
          
          {feedbackMutation.isPending && (
            <div className="flex items-center justify-center p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              스타일 학습 중...
            </div>
          )}
        </div>
      )}
    </div>
  );
}