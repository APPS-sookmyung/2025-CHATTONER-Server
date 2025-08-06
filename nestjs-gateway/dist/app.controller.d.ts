import { ConversionRequestDto } from './dto/conversion-request.dto';
import { ConversionResponseDto } from './dto/conversion-response.dto';
import { FeedbackRequestDto } from './dto/feedback-request.dto';
import { FeedbackResponseDto } from './dto/feedback-response.dto';
export declare class AppController {
    getRoot(): string;
    convertText(body: ConversionRequestDto): Promise<ConversionResponseDto>;
    submitFeedback(body: FeedbackRequestDto): Promise<FeedbackResponseDto>;
}
