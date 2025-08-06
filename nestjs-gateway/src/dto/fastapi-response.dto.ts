// nestjs-gateway/src/dto/fastapi-response.dto.ts
import { IsString, IsNumber, IsArray } from 'class-validator';

export class FastApiResponseDto {
    @IsString()
    answer!: string;

    @IsArray()
    sources!: string[]; // 출처 문서 URL 또는 제목

    @IsNumber()
    confidence!: number; // 예: 0.87 (확신 정도)
}
