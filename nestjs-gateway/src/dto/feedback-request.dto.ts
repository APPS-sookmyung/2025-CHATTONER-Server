// nestjs-gateway/src/dto/feedback-request.dto.ts
import { IsString, IsOptional } from 'class-validator';

export class FeedbackRequestDto {
    @IsString()
    feedback_text: string;

    @IsOptional()
    user_profile?: Record<string, any>;
}
