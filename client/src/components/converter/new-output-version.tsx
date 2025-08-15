import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Copy, ThumbsUp, ThumbsDown, MessageSquare } from 'lucide-react';
import { Textarea } from '@/components/ui/textarea';
import { useToast } from '@/hooks/use-toast';

interface NewOutputVersionProps {
  type: 'direct' | 'gentle' | 'neutral';
  label: string;
  description: string;
  text: string;
  conversionId?: number;
  onSelect: (type: 'direct' | 'gentle' | 'neutral', feedback?: string) => void;
  onCopy: () => void;
}

const typeStyles = {
  direct: 'border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-950',
  gentle: 'border-blue-200 bg-blue-50 dark:border-blue-800 dark:bg-blue-950',
  neutral: 'border-gray-200 bg-gray-50 dark:border-gray-800 dark:bg-gray-950',
};

const typeBadges = {
  direct: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
  gentle: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
  neutral: 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200',
};

export default function NewOutputVersion({
  type,
  label,
  description,
  text,
  conversionId,
  onSelect,
  onCopy,
}: NewOutputVersionProps) {
  const [showFeedback, setShowFeedback] = useState(false);
  const [feedback, setFeedback] = useState('');
  const [isSelected, setIsSelected] = useState(false);
  const { toast } = useToast();

  const handleSelect = () => {
    setIsSelected(true);
    onSelect(type, feedback);
    toast({
      title: '선택 완료',
      description: `${label} 버전을 선택했습니다. 스타일이 업데이트됩니다.`,
    });
  };

  const handleCopyAndSelect = () => {
    onCopy();
    handleSelect();
  };

  return (
    <Card className={`${typeStyles[type]} transition-all duration-200 ${isSelected ? 'ring-2 ring-primary' : ''}`}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">{label}</CardTitle>
          <Badge variant="outline" className={typeBadges[type]}>
            {type}
          </Badge>
        </div>
        <p className="text-sm text-muted-foreground">{description}</p>
      </CardHeader>
      
      <CardContent className="pb-4">
        <div className="p-4 rounded-lg bg-white dark:bg-gray-800 border">
          <p className="text-sm leading-relaxed whitespace-pre-wrap">{text}</p>
        </div>
      </CardContent>
      
      <CardFooter className="flex flex-col gap-3 pt-0">
        <div className="flex gap-2 w-full">
          <Button
            variant="outline"
            size="sm"
            onClick={onCopy}
            className="flex-1"
          >
            <Copy className="w-4 h-4 mr-2" />
            복사
          </Button>
          <Button
            variant={isSelected ? "default" : "outline"}
            size="sm"
            onClick={handleSelect}
            className="flex-1"
            disabled={isSelected}
          >
            <ThumbsUp className="w-4 h-4 mr-2" />
            {isSelected ? '선택됨' : '이걸로 선택'}
          </Button>
        </div>
        
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setShowFeedback(!showFeedback)}
          className="w-full text-xs"
        >
          <MessageSquare className="w-3 h-3 mr-2" />
          피드백 주기
        </Button>
        
        {showFeedback && (
          <div className="w-full space-y-2">
            <Textarea
              placeholder="이 버전에 대한 피드백을 남겨주세요. (예: 좀 더 부드럽게, 더 간결하게 등)"
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
              className="text-xs"
              rows={2}
            />
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                onSelect(type, feedback);
                setFeedback('');
                setShowFeedback(false);
                toast({
                  title: '피드백 전송',
                  description: '피드백이 반영되어 다음 변환에 적용됩니다.',
                });
              }}
              className="w-full text-xs"
            >
              피드백 전송
            </Button>
          </div>
        )}
      </CardFooter>
    </Card>
  );
}