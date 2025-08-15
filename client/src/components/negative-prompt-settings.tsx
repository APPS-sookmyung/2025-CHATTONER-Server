import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Settings, Filter, Zap } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface UserNegativePreferences {
  rhetoricLevel: 'strict' | 'moderate' | 'lenient';
  repetitionTolerance: 'strict' | 'moderate' | 'lenient';
  punctuationStyle: 'minimal' | 'standard' | 'verbose';
  contentFocus: 'content' | 'balanced' | 'format';
  bulletPreference: 'avoid' | 'minimal' | 'prefer';
  emoticonPolicy: 'none' | 'minimal' | 'contextual' | 'frequent';
}

interface NegativePromptSettingsProps {
  userId: string;
  onSave?: (preferences: UserNegativePreferences) => void;
}

export function NegativePromptSettings({ userId, onSave }: NegativePromptSettingsProps) {
  const [preferences, setPreferences] = useState<UserNegativePreferences>({
    rhetoricLevel: 'moderate',
    repetitionTolerance: 'moderate',
    punctuationStyle: 'standard',
    contentFocus: 'balanced',
    bulletPreference: 'minimal',
    emoticonPolicy: 'contextual'
  });
  
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();

  const rhetoricOptions = [
    { value: 'strict', label: '엄격', description: '모든 과장 표현 금지' },
    { value: 'moderate', label: '보통', description: '적당한 수준의 표현' },
    { value: 'lenient', label: '관대', description: '자연스러운 표현 허용' }
  ];

  const repetitionOptions = [
    { value: 'strict', label: '엄격', description: '반복 표현 완전 금지' },
    { value: 'moderate', label: '보통', description: '필요시에만 허용' },
    { value: 'lenient', label: '관대', description: '자연스러운 반복 허용' }
  ];

  const punctuationOptions = [
    { value: 'minimal', label: '최소', description: '꼭 필요한 쉼표만' },
    { value: 'standard', label: '표준', description: '일반적인 사용' },
    { value: 'verbose', label: '상세', description: '풍부한 쉼표 사용' }
  ];

  const contentFocusOptions = [
    { value: 'content', label: '내용 중심', description: '핵심 내용 우선' },
    { value: 'balanced', label: '균형', description: '내용과 형식 조화' },
    { value: 'format', label: '형식 중심', description: '체계적 구성 우선' }
  ];

  const bulletOptions = [
    { value: 'avoid', label: '회피', description: '불렛 포인트 사용 안함' },
    { value: 'minimal', label: '최소', description: '필요시에만 사용' },
    { value: 'prefer', label: '선호', description: '적극적 사용' }
  ];

  const emoticonOptions = [
    { value: 'none', label: '없음', description: '이모티콘 완전 금지' },
    { value: 'minimal', label: '최소', description: '기본적인 것만' },
    { value: 'contextual', label: '상황적', description: '맥락에 맞게' },
    { value: 'frequent', label: '빈번', description: '자주 사용' }
  ];

  const handleSave = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/negative-preferences', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          userId,
          preferences
        }),
      });

      if (response.ok) {
        toast({
          title: "설정 저장 완료",
          description: "네거티브 프롬프트 선호도가 저장되었습니다.",
        });
        onSave?.(preferences);
      } else {
        throw new Error('저장 실패');
      }
    } catch (error) {
      toast({
        title: "저장 실패",
        description: "설정 저장 중 오류가 발생했습니다.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const updatePreference = <K extends keyof UserNegativePreferences>(
    key: K,
    value: UserNegativePreferences[K]
  ) => {
    setPreferences(prev => ({ ...prev, [key]: value }));
  };

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <div className="flex items-center gap-2">
          <Filter className="h-5 w-5" />
          <CardTitle>네거티브 프롬프트 설정</CardTitle>
        </div>
        <CardDescription>
          텍스트 변환 시 제외하고 싶은 표현들을 맞춤 설정하세요. 
          이 설정은 모든 텍스트 변환에 적용됩니다.
        </CardDescription>
      </CardHeader>
      
      <CardContent className="space-y-6">
        {/* 미사여구 설정 */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Badge variant="outline">🎭</Badge>
            <Label className="text-base font-medium">미사여구 제한</Label>
          </div>
          <p className="text-sm text-muted-foreground">
            "놀라운", "대단한", "훌륭한" 같은 과장된 수식어 사용을 어느 정도까지 허용할지 설정합니다.
          </p>
          <Select
            value={preferences.rhetoricLevel}
            onValueChange={(value: 'strict' | 'moderate' | 'lenient') => 
              updatePreference('rhetoricLevel', value)
            }
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {rhetoricOptions.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  <div className="flex flex-col">
                    <span>{option.label}</span>
                    <span className="text-xs text-muted-foreground">{option.description}</span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <Separator />

        {/* 반복 표현 설정 */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Badge variant="outline">🔁</Badge>
            <Label className="text-base font-medium">반복 표현 제한</Label>
          </div>
          <p className="text-sm text-muted-foreground">
            "그리고 또한", "~에 대해서도" 같은 비슷한 단어들의 반복 사용을 제한합니다.
          </p>
          <Select
            value={preferences.repetitionTolerance}
            onValueChange={(value: 'strict' | 'moderate' | 'lenient') => 
              updatePreference('repetitionTolerance', value)
            }
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {repetitionOptions.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  <div className="flex flex-col">
                    <span>{option.label}</span>
                    <span className="text-xs text-muted-foreground">{option.description}</span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <Separator />

        {/* 쉼표 사용 스타일 */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Badge variant="outline">✏️</Badge>
            <Label className="text-base font-medium">쉼표 사용 스타일</Label>
          </div>
          <p className="text-sm text-muted-foreground">
            문장에서 쉼표를 어느 정도까지 사용할지 설정합니다.
          </p>
          <Select
            value={preferences.punctuationStyle}
            onValueChange={(value: 'minimal' | 'standard' | 'verbose') => 
              updatePreference('punctuationStyle', value)
            }
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {punctuationOptions.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  <div className="flex flex-col">
                    <span>{option.label}</span>
                    <span className="text-xs text-muted-foreground">{option.description}</span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <Separator />

        {/* 내용 vs 형식 우선순위 */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Badge variant="outline">📝</Badge>
            <Label className="text-base font-medium">내용 vs 형식 우선순위</Label>
          </div>
          <p className="text-sm text-muted-foreground">
            텍스트 변환 시 내용을 우선할지, 형식을 우선할지 설정합니다.
          </p>
          <Select
            value={preferences.contentFocus}
            onValueChange={(value: 'content' | 'balanced' | 'format') => 
              updatePreference('contentFocus', value)
            }
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {contentFocusOptions.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  <div className="flex flex-col">
                    <span>{option.label}</span>
                    <span className="text-xs text-muted-foreground">{option.description}</span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <Separator />

        {/* 불렛 포인트 사용 */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Badge variant="outline">📋</Badge>
            <Label className="text-base font-medium">불렛 포인트 사용</Label>
          </div>
          <p className="text-sm text-muted-foreground">
            "•", "-", "1." 같은 불렛 포인트 사용을 설정합니다.
          </p>
          <Select
            value={preferences.bulletPreference}
            onValueChange={(value: 'avoid' | 'minimal' | 'prefer') => 
              updatePreference('bulletPreference', value)
            }
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {bulletOptions.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  <div className="flex flex-col">
                    <span>{option.label}</span>
                    <span className="text-xs text-muted-foreground">{option.description}</span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <Separator />

        {/* 이모티콘 정책 */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Badge variant="outline">😊</Badge>
            <Label className="text-base font-medium">이모티콘 사용 정책</Label>
          </div>
          <p className="text-sm text-muted-foreground">
            텍스트에서 이모티콘 사용을 어느 정도까지 허용할지 설정합니다.
          </p>
          <Select
            value={preferences.emoticonPolicy}
            onValueChange={(value: 'none' | 'minimal' | 'contextual' | 'frequent') => 
              updatePreference('emoticonPolicy', value)
            }
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {emoticonOptions.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  <div className="flex flex-col">
                    <span>{option.label}</span>
                    <span className="text-xs text-muted-foreground">{option.description}</span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <Separator />

        {/* 저장 버튼 */}
        <div className="flex justify-end gap-3">
          <Button
            onClick={handleSave}
            disabled={isLoading}
            className="flex items-center gap-2"
          >
            <Zap className="h-4 w-4" />
            {isLoading ? "저장 중..." : "설정 저장"}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}