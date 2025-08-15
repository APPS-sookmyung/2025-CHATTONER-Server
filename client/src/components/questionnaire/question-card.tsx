import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { ChevronLeft, ChevronRight, Plus, Info } from "lucide-react";
import type { Question } from "@/data/questions";

interface QuestionCardProps {
  question: Question;
  selectedOptions: string[];
  customInput: string;
  onAnswerChange: (questionId: string, selectedOptions: string[], customInput?: string) => void;
  onNext: () => void;
  onPrevious: () => void;
  onSkip: () => void;
  canGoBack: boolean;
  isLastQuestion: boolean;
}

export default function QuestionCard({
  question,
  selectedOptions,
  customInput,
  onAnswerChange,
  onNext,
  onPrevious,
  onSkip,
  canGoBack,
  isLastQuestion
}: QuestionCardProps) {
  const [localCustomInput, setLocalCustomInput] = useState(customInput);

  const handleOptionChange = (optionValue: string, checked: boolean) => {
    let newSelected: string[];
    
    if (question.allowMultiple) {
      if (checked) {
        newSelected = [...selectedOptions, optionValue];
      } else {
        newSelected = selectedOptions.filter(opt => opt !== optionValue);
      }
    } else {
      newSelected = checked ? [optionValue] : [];
    }
    
    onAnswerChange(question.id, newSelected, localCustomInput);
  };

  const handleCustomInputChange = (value: string) => {
    setLocalCustomInput(value);
    onAnswerChange(question.id, selectedOptions, value);
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'situational':
        return 'bg-primary/10 text-primary';
      case 'preferences':
        return 'bg-secondary/10 text-secondary';
      case 'expressions':
        return 'bg-accent/10 text-accent';
      default:
        return 'bg-gray-100 text-gray-600';
    }
  };

  const hasAnswer = selectedOptions.length > 0 || localCustomInput.trim();

  return (
    <Card className="shadow-sm border border-gray-200">
      <CardContent className="p-8">
        <div className="mb-6">
          <div className="flex items-center space-x-2 mb-3">
            <Badge className={`${getCategoryColor(question.category)} font-medium`}>
              {question.categoryLabel}
            </Badge>
            {question.allowMultiple && (
              <div className="flex items-center space-x-1 text-gray-400">
                <Info className="w-4 h-4" />
                <span className="text-xs">복수 선택 가능</span>
              </div>
            )}
          </div>
          <h3 className="text-xl font-semibold text-gray-900 mb-2">{question.question}</h3>
          <p className="text-gray-600">{question.description}</p>
        </div>

        {/* Answer Options */}
        <div className="space-y-3 mb-6">
          {question.allowMultiple ? (
            // Multiple choice with checkboxes
            <div className="space-y-3">
              {question.options.map((option, index) => (
                <div key={index} className="group">
                  <label className={`flex items-center p-4 border-2 rounded-xl cursor-pointer transition-all duration-200 ${
                    selectedOptions.includes(option) 
                      ? 'border-primary bg-primary/5' 
                      : 'border-gray-200 hover:border-primary/30 hover:bg-primary/5'
                  }`}>
                    <Checkbox
                      checked={selectedOptions.includes(option)}
                      onCheckedChange={(checked) => handleOptionChange(option, checked as boolean)}
                      className="mr-4"
                    />
                    <span className="text-gray-700 font-medium flex-1">{option}</span>
                    {selectedOptions.includes(option) && (
                      <div className="text-primary">
                        <ChevronRight className="w-4 h-4" />
                      </div>
                    )}
                  </label>
                </div>
              ))}
            </div>
          ) : (
            // Single choice with radio buttons
            <RadioGroup 
              value={selectedOptions[0] || ''} 
              onValueChange={(value) => handleOptionChange(value, true)}
              className="space-y-3"
            >
              {question.options.map((option, index) => (
                <div key={index} className="group">
                  <label className={`flex items-center p-4 border-2 rounded-xl cursor-pointer transition-all duration-200 ${
                    selectedOptions.includes(option) 
                      ? 'border-primary bg-primary/5' 
                      : 'border-gray-200 hover:border-primary/30 hover:bg-primary/5'
                  }`}>
                    <RadioGroupItem value={option} className="mr-4" />
                    <span className="text-gray-700 font-medium flex-1">{option}</span>
                    {selectedOptions.includes(option) && (
                      <div className="text-primary">
                        <ChevronRight className="w-4 h-4" />
                      </div>
                    )}
                  </label>
                </div>
              ))}
            </RadioGroup>
          )}
        </div>

        {/* Custom Input Option */}
        {question.allowCustom && (
          <div className="mb-6 p-4 bg-gray-50 rounded-xl">
            <div className="flex items-center space-x-2 mb-2">
              <Plus className="w-4 h-4 text-gray-400" />
              <span className="text-sm font-medium text-gray-700">직접 입력</span>
            </div>
            <Input
              value={localCustomInput}
              onChange={(e) => handleCustomInputChange(e.target.value)}
              placeholder="자주 사용하는 다른 표현이 있나요?"
              className="border-gray-200 focus:ring-2 focus:ring-primary focus:border-transparent"
            />
          </div>
        )}

        {/* Navigation Buttons */}
        <div className="flex justify-between items-center">
          <Button
            variant="ghost"
            onClick={onPrevious}
            disabled={!canGoBack}
            className="flex items-center space-x-2"
          >
            <ChevronLeft className="w-4 h-4" />
            <span>이전</span>
          </Button>

          <div className="flex space-x-3">
            <Button
              variant="ghost"
              onClick={onSkip}
              className="text-gray-600 hover:text-gray-800"
            >
              건너뛰기
            </Button>
            <Button
              onClick={() => {
                console.log('버튼 클릭됨:', isLastQuestion ? '완료' : '다음');
                onNext();
              }}
              disabled={false} // Allow progression even without answers
              className="flex items-center space-x-2 px-8"
            >
              <span>{isLastQuestion ? '완료' : '다음'}</span>
              <ChevronRight className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
