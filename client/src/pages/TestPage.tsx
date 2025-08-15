import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';

export default function TestPage() {
  const [apiResponse, setApiResponse] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [testData, setTestData] = useState({
    userId: 'test-user-001',
    inputText: '안녕하세요. 오늘 회의에서 논의할 안건을 정리해주세요.',
    context: 'business',
    feedbackText: '더 친근하게 해주세요',
    rating: 4,
    selectedVersion: 'gentle'
  });

  const testAPI = async (endpoint: string, method: string = 'GET', body?: any) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`/api${endpoint}`, {
        method,
        headers: {
          'Content-Type': 'application/json',
        },
        body: body ? JSON.stringify(body) : undefined,
      });
      
      const data = await response.json();
      setApiResponse({ endpoint, status: response.status, data });
      
      if (!response.ok) {
        setError(`API 오류 (${response.status}): ${data.detail || data.error || '알 수 없는 오류'}`);
      }
    } catch (err) {
      setError(`연결 오류: ${err.message}`);
      setApiResponse({ endpoint, error: err.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="text-center">
        <h1 className="text-3xl font-bold mb-2">🧪 Chat Toner API 테스트</h1>
        <p className="text-gray-600">프론트엔드와 백엔드 연결 상태를 확인하고 API를 테스트하세요</p>
      </div>

      {/* 기본 연결 테스트 */}
      <Card>
        <CardHeader>
          <CardTitle>🔍 연결 상태 확인</CardTitle>
          <CardDescription>백엔드 서버 상태를 확인합니다</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-4">
            <Button onClick={() => testAPI('/health')} disabled={loading}>
              Health Check
            </Button>
            <Button onClick={() => testAPI('/negative-preferences/test-user-001')} disabled={loading}>
              사용자 선호도 조회
            </Button>
            <Button onClick={() => testAPI('/profile/test-user-001')} disabled={loading}>
              프로필 조회
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* 텍스트 변환 테스트 */}
      <Card>
        <CardHeader>
          <CardTitle>🎯 텍스트 변환 테스트</CardTitle>
          <CardDescription>실제 텍스트 스타일 변환을 테스트합니다</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">사용자 ID</label>
              <Input 
                value={testData.userId}
                onChange={(e) => setTestData({...testData, userId: e.target.value})}
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">컨텍스트</label>
              <select 
                className="w-full p-2 border rounded"
                value={testData.context}
                onChange={(e) => setTestData({...testData, context: e.target.value})}
              >
                <option value="business">업무</option>
                <option value="personal">개인적</option>
                <option value="report">보고서</option>
              </select>
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-2">변환할 텍스트</label>
            <Textarea 
              value={testData.inputText}
              onChange={(e) => setTestData({...testData, inputText: e.target.value})}
              rows={3}
            />
          </div>
          
          <Button 
            onClick={() => testAPI('/convert', 'POST', {
              userId: testData.userId,
              inputText: testData.inputText,
              context: testData.context
            })}
            disabled={loading}
            className="w-full"
          >
            {loading ? '변환 중...' : '텍스트 변환하기'}
          </Button>
        </CardContent>
      </Card>

      {/* 피드백 테스트 */}
      <Card>
        <CardHeader>
          <CardTitle>💬 피드백 처리 테스트</CardTitle>
          <CardDescription>사용자 피드백 처리 및 프로필 업데이트를 테스트합니다</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">피드백 내용</label>
              <Input 
                value={testData.feedbackText}
                onChange={(e) => setTestData({...testData, feedbackText: e.target.value})}
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">평점 (1-5)</label>
              <Input 
                type="number"
                min="1"
                max="5"
                value={testData.rating}
                onChange={(e) => setTestData({...testData, rating: parseInt(e.target.value)})}
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">선택된 버전</label>
              <select 
                className="w-full p-2 border rounded"
                value={testData.selectedVersion}
                onChange={(e) => setTestData({...testData, selectedVersion: e.target.value})}
              >
                <option value="direct">직접적</option>
                <option value="gentle">부드러운</option>
                <option value="neutral">중립적</option>
              </select>
            </div>
          </div>
          
          <Button 
            onClick={() => testAPI('/feedback', 'POST', {
              userId: testData.userId,
              feedbackText: testData.feedbackText,
              rating: testData.rating,
              selectedVersion: testData.selectedVersion,
              originalText: testData.inputText,
              convertedText: "변환된 텍스트 예시"
            })}
            disabled={loading}
            className="w-full"
          >
            {loading ? '처리 중...' : '피드백 보내기'}
          </Button>
        </CardContent>
      </Card>

      {/* 응답 결과 */}
      {error && (
        <Alert className="border-red-200">
          <AlertDescription className="text-red-700">
            {error}
          </AlertDescription>
        </Alert>
      )}

      {apiResponse && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              📋 API 응답 결과
              <Badge variant={apiResponse.status < 400 ? "default" : "destructive"}>
                {apiResponse.status || 'ERROR'}
              </Badge>
            </CardTitle>
            <CardDescription>
              엔드포인트: <code>{apiResponse.endpoint}</code>
            </CardDescription>
          </CardHeader>
          <CardContent>
            <pre className="bg-gray-100 p-4 rounded-lg overflow-auto text-sm">
              {JSON.stringify(apiResponse.data || apiResponse.error, null, 2)}
            </pre>
          </CardContent>
        </Card>
      )}

      {loading && (
        <div className="text-center py-8">
          <div className="animate-spin inline-block w-8 h-8 border-4 border-current border-t-transparent text-blue-600 rounded-full"></div>
          <p className="mt-2 text-gray-600">API 요청 처리 중...</p>
        </div>
      )}
    </div>
  );
}