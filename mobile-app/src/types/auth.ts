export interface User {
  id: number;
  full_name: string;
  email: string;
  phone_number?: string;
  is_admin: boolean;
  created_at: string;
}

export interface AuthResponse {
  user: User;
  tokens: {
    access: string;
    refresh: string;
  };
}
