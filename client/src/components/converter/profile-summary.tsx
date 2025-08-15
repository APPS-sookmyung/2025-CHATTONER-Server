import { Card, CardContent } from "@/components/ui/card";
import { User } from "lucide-react";
import { analyzeUserProfile } from "@/lib/prompt-engineering";
import type { UserProfile } from "@shared/schema";

interface ProfileSummaryProps {
  userProfile: UserProfile;
}

export default function ProfileSummary({ userProfile }: ProfileSummaryProps) {
  const analysis = analyzeUserProfile(userProfile);

  const getProgressWidth = (level: number) => `${level * 10}%`;

  return (
    <Card className="bg-gradient-to-r from-primary/5 to-secondary/5 border-primary/20">
      <CardContent className="p-6">
        <div className="flex items-center space-x-3 mb-4">
          <div className="w-10 h-10 bg-primary rounded-xl flex items-center justify-center">
            <User className="text-white w-5 h-5" />
          </div>
          <div>
            <h3 className="font-semibold text-gray-900">당신의 말투 프로필</h3>
            <p className="text-sm text-gray-600">수집된 선호도를 기반으로 텍스트를 변환합니다</p>
          </div>
        </div>
        
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-white/60 rounded-lg p-4">
            <div className="text-sm text-gray-600 mb-1">정중함 정도</div>
            <div className="flex items-center space-x-2">
              <div className="flex-1 bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-primary h-2 rounded-full transition-all duration-500" 
                  style={{ width: getProgressWidth(userProfile.formalityLevel) }}
                />
              </div>
              <span className="text-sm font-medium text-gray-700">
                {userProfile.formalityLevel}/10
              </span>
            </div>
          </div>
          
          <div className="bg-white/60 rounded-lg p-4">
            <div className="text-sm text-gray-600 mb-1">친근함 정도</div>
            <div className="flex items-center space-x-2">
              <div className="flex-1 bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-secondary h-2 rounded-full transition-all duration-500" 
                  style={{ width: getProgressWidth(userProfile.friendlinessLevel) }}
                />
              </div>
              <span className="text-sm font-medium text-gray-700">
                {userProfile.friendlinessLevel}/10
              </span>
            </div>
          </div>
          
          <div className="bg-white/60 rounded-lg p-4">
            <div className="text-sm text-gray-600 mb-1">감정 표현</div>
            <div className="flex items-center space-x-2">
              <div className="flex-1 bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-accent h-2 rounded-full transition-all duration-500" 
                  style={{ width: getProgressWidth(userProfile.emotionLevel) }}
                />
              </div>
              <span className="text-sm font-medium text-gray-700">
                {userProfile.emotionLevel}/10
              </span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
