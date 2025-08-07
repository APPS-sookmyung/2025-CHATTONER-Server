// nestjs-gateway/src/dto/query-to-fastapi.dto.ts
import { IsString, IsOptional, IsNumber } from 'class-validator';

export class QueryToFastApiDto {
    @IsString()
    query!: string;

    @IsOptional()
    @IsNumber()
    top_k?: number; // 유사도 검색 상위 몇 개까지 추출할지
}
