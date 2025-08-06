// nestjs-gateway/src/dto/conversion-request.dto.ts
import { IsString, IsOptional, IsArray } from 'class-validator';
import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';

export class ConversionRequestDto {
    @ApiProperty({ description: '변환할 원본 텍스트' })
    @IsString()
    text: string;

    @ApiPropertyOptional({ description: '텍스트가 사용되는 맥락 (선택사항)' })
    @IsOptional()
    @IsString()
    context?: string;

    @ApiPropertyOptional({ description: '사용자 프로필 정보 (선택사항)', type: Object })
    @IsOptional()
    user_profile?: Record<string, any>;

    @ApiPropertyOptional({ description: '선호하지 않는 스타일 목록 (선택사항)', type: [String] })
    @IsOptional()
    @IsArray()
    negative_preferences?: string[];
}
