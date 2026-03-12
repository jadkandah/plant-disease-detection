import React, { useState, useContext } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, ActivityIndicator, ScrollView } from 'react-native';
import { AuthContext } from '../../store/AuthContext';
import { useTranslation } from '../../store/LanguageContext';
import apiClient from '../../services/auth/apiClient';

export const SignUpScreen = ({ navigation }: { navigation: any }) => {
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [phoneNumber, setPhoneNumber] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { login } = useContext(AuthContext);
  const { t, isRTL } = useTranslation();

  const handleSignUp = async () => {
    if (!fullName || !email || !password || !phoneNumber) {
      setError(t('auth.fillAllFields'));
      return;
    }
    setLoading(true);
    setError('');

    try {
      const response = await apiClient.post('/auth/register/', {
        full_name: fullName, email, phone_number: phoneNumber, password,
      });
      await login(response.data);
    } catch (err: any) {
      console.log(err.response?.data);
      setError(err.response?.data?.email?.[0] || t('auth.registrationFailed'));
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
