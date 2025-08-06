import { Body, Controller, HttpException, HttpStatus, Post, Get } from '@nestjs/common';
import { ConversionRequestDto } from './dto/conversion-request.dto';
import { ConversionResponseDto } from './dto/conversion-response.dto';
import { FeedbackRequestDto } from './dto/feedback-request.dto';
import { FeedbackResponseDto } from './dto/feedback-response.dto';

@Controller()
export class AppController {
  @Get()
  getRoot(): string {
    return '서버가 정상 작동 중입니다!';
  }

  @Post('convert')
  async convertText(
    @Body() body: ConversionRequestDto
  ): Promise<ConversionResponseDto> {
    try {
      // 💡 여기에 FastAPI 연동 또는 내부 변환 로직 연결 예정
      const convertedText = `변환된 텍스트: ${body.text}`; // 임시 결과
      return {
        converted_text: convertedText,
      };
    } catch (e) {
      throw new HttpException('텍스트 변환 중 오류 발생', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }

  @Post('feedback')
  async submitFeedback(
    @Body() body: FeedbackRequestDto
  ): Promise<FeedbackResponseDto> {
    try {
      // 💡 여기에 피드백 저장 또는 전송 로직 연결 예정
      return {
        success: true,
        message: '피드백이 반영되었습니다.',
        data: {
          received_feedback: body.feedback_text,
        },
      };
    } catch (e) {
      throw new HttpException('피드백 처리 중 오류 발생', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }
}
