import React, { useState, useContext } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, ActivityIndicator, ScrollView } from 'react-native';
import { AuthContext } from '../../store/AuthContext';
import { useTranslation } from '../../store/LanguageContext';
import apiClient from '../../services/auth/apiClient';

const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;
const PHONE_PATTERN = /^07\d{8}$/;

const isStrongPassword = (value: string) =>
  value.length >= 8 &&
  /[A-Z]/.test(value) &&
  /[a-z]/.test(value) &&
  /\d/.test(value) &&
  /[^A-Za-z0-9]/.test(value);

export const SignUpScreen = ({ navigation }: { navigation: any }) => {
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [phoneNumber, setPhoneNumber] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { login } = useContext(AuthContext);
  const { t, isRTL } = useTranslation();

  const getServerError = (data: any) => {
    if (!data) return t('auth.registrationFailed');
    if (typeof data === 'string') return data;

    const fields = ['full_name', 'email', 'phone_number', 'password', 'non_field_errors', 'detail'];
    for (const field of fields) {
      const value = data[field];
      if (Array.isArray(value) && value.length > 0) return String(value[0]);
      if (typeof value === 'string') return value;
    }

    return t('auth.registrationFailed');
  };

  const validateForm = () => {
    const normalizedFullName = fullName.trim().replace(/\s+/g, ' ');
    const normalizedEmail = email.trim().toLowerCase();
    const normalizedPhoneNumber = phoneNumber.trim().replace(/[\s-]/g, '');

    if (!normalizedFullName || !normalizedEmail || !password || !normalizedPhoneNumber) {
      setError(t('auth.fillAllFields'));
      return null;
    }

    if (normalizedFullName.length < 2) {
      setError(t('auth.invalidFullName'));
      return null;
    }

    if (!EMAIL_PATTERN.test(normalizedEmail)) {
      setError(t('auth.invalidEmail'));
      return null;
    }

    if (!PHONE_PATTERN.test(normalizedPhoneNumber)) {
      setError(t('auth.invalidPhone'));
      return null;
    }

    if (!isStrongPassword(password)) {
      setError(t('auth.weakPassword'));
      return null;
    }

    const passwordLower = password.toLowerCase();
    const emailLocalPart = normalizedEmail.split('@')[0];
    const nameParts = normalizedFullName.toLowerCase().split(' ').filter((part) => part.length >= 3);
    const containsPersonalInfo = [emailLocalPart, ...nameParts].some((part) => part && passwordLower.includes(part));

    if (containsPersonalInfo) {
      setError(t('auth.passwordContainsPersonalInfo'));
      return null;
    }

    return {
      full_name: normalizedFullName,
      email: normalizedEmail,
      phone_number: normalizedPhoneNumber,
      password,
    };
  };

  const handleSignUp = async () => {
    const payload = validateForm();
    if (!payload) return;

    setLoading(true);
    setError('');

    try {
      const response = await apiClient.post('/auth/register/', payload);
      await login(response.data);
    } catch (err: any) {
      console.log(err.response?.data);
      setError(getServerError(err.response?.data));
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={[styles.title, isRTL && styles.rtlText]}>{t('auth.createAccount')}</Text>

      {error ? <Text style={styles.errorText}>{error}</Text> : null}

      <TextInput style={[styles.input, isRTL && styles.rtlInput]} placeholder={t('auth.fullName')} value={fullName} onChangeText={setFullName} textAlign={isRTL ? 'right' : 'left'} />
      <TextInput style={[styles.input, isRTL && styles.rtlInput]} placeholder={t('auth.email')} keyboardType="email-address" autoCapitalize="none" value={email} onChangeText={setEmail} textAlign={isRTL ? 'right' : 'left'} />
      <TextInput style={[styles.input, isRTL && styles.rtlInput]} placeholder={t('auth.phoneNumber')} keyboardType="phone-pad" value={phoneNumber} onChangeText={setPhoneNumber} textAlign={isRTL ? 'right' : 'left'} />
      <TextInput style={[styles.input, isRTL && styles.rtlInput]} placeholder={t('auth.password')} secureTextEntry value={password} onChangeText={setPassword} textAlign={isRTL ? 'right' : 'left'} />

      <TouchableOpacity style={styles.button} onPress={handleSignUp} disabled={loading}>
        {loading ? <ActivityIndicator color="#fff" /> : <Text style={styles.buttonText}>{t('auth.signUp')}</Text>}
      </TouchableOpacity>

      <TouchableOpacity onPress={() => navigation.goBack()} style={styles.linkContainer}>
        <Text style={styles.linkText}>{t('auth.hasAccount')}</Text>
      </TouchableOpacity>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: { flexGrow: 1, padding: 20, justifyContent: 'center', backgroundColor: '#f9f9f9' },
  title: { fontSize: 28, fontWeight: 'bold', color: '#2E7D32', marginBottom: 30, textAlign: 'center' },
  input: { backgroundColor: '#fff', borderWidth: 1, borderColor: '#ccc', padding: 15, borderRadius: 8, marginBottom: 15, fontSize: 16 },
  rtlInput: { textAlign: 'right' },
  rtlText: { textAlign: 'right' },
  button: { backgroundColor: '#2E7D32', padding: 15, borderRadius: 8, alignItems: 'center', marginTop: 10 },
  buttonText: { color: '#fff', fontSize: 18, fontWeight: 'bold' },
  linkContainer: { marginTop: 20, alignItems: 'center' },
  linkText: { color: '#2E7D32', fontSize: 16 },
  errorText: { color: '#d32f2f', marginBottom: 15, textAlign: 'center' },
});
