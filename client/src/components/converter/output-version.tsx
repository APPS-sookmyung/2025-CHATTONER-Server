import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ThumbsUp, ThumbsDown, Copy, Check } from "lucide-react";

interface OutputVersionProps {
  type: 'standard' | 'casual' | 'formal';
  label: string;
  description: string;
  text: string;
  onLike: () => void;
  onDislike: () => void;
  onCopy: () => void;
}

export default function OutputVersion({
  type,
  label,
  description,
  text,
  onLike,
  onDislike,
  onCopy
}: OutputVersionProps) {
  const [feedback, setFeedback] = useState<'like' | 'dislike' | null>(null);
  const [copied, setCopied] = useState(false);

  const handleLike = () => {
    setFeedback('like');
    onLike();
  };

  const handleDislike = () => {
    setFeedback('dislike');
    onDislike();
  };

  const handleCopy = () => {
    setCopied(true);
    onCopy();
    setTimeout(() => setCopied(false), 1000);
  };

  const getBadgeColor = (type: string) => {
    switch (type) {
      case 'standard':
        return 'bg-primary/10 text-primary';
      case 'casual':
        return 'bg-secondary/10 text-secondary';
      case 'formal':
        return 'bg-accent/10 text-accent';
      default:
        return 'bg-gray-100 text-gray-600';
    }
  };

  const getBorderColor = (type: string) => {
    switch (type) {
      case 'standard':
        return 'hover:border-primary/30';
      case 'casual':
        return 'hover:border-secondary/30';
      case 'formal':
        return 'hover:border-accent/30';
      default:
        return 'hover:border-gray-300';
    }
  };

  return (
    <div className={`border border-gray-200 rounded-xl p-4 transition-colors ${getBorderColor(type)}`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <Badge className={`${getBadgeColor(type)} font-medium`}>
            {label}
          </Badge>
          <span className="text-sm text-gray-600">{description}</span>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleLike}
            className={`p-1 ${feedback === 'like' ? 'text-success' : 'text-gray-400 hover:text-success'}`}
            title="좋아요"
          >
            <ThumbsUp className="w-4 h-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleDislike}
            className={`p-1 ${feedback === 'dislike' ? 'text-error' : 'text-gray-400 hover:text-error'}`}
            title="별로예요"
          >
            <ThumbsDown className="w-4 h-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCopy}
            className="p-1 text-gray-400 hover:text-primary"
            title="복사"
          >
            {copied ? <Check className="w-4 h-4 text-success" /> : <Copy className="w-4 h-4" />}
          </Button>
        </div>
      </div>
      <p className="text-gray-800 leading-relaxed">{text}</p>
    </div>
  );
}
