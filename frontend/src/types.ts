export interface User {
  id: string;
  email: string;
  role: string;
  created_at: string;
}

export interface Analysis {
  id: string;
  input_text: string;
  results: any;
  created_at: string;
}

export interface Book {
  id: string;
  filename: string;
  philosopher: string;
  status: string;
  page_count?: number;
  chunk_count: number;
  uploaded_at: string;
  processed_at?: string;
}

export interface Experiment {
  id: string;
  module_name: string;
  algorithm: string;
  accuracy?: number;
  f1_score?: number;
  rmse?: number;
  params_json: any;
  training_size?: number;
  test_size?: number;
  run_at: string;
  notes?: string;
}

export interface ModuleResult {
  [key: string]: any;
}

export interface AnalyzeRequest {
  text: string;
  modules?: string[];
}

export interface AnalyzeResponse {
  results: { [module: string]: any };
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  password: string;
}

export interface TokenResponse {
  token: string;
  token_type: string;
}

export interface QueryRequest {
  message: string;
  history: { role: string; content: string }[];
  context: any;
}