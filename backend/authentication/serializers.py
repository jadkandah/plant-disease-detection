import re

from django.contrib.auth import authenticate
from django.contrib.auth.password_validation import validate_password as django_validate_password
from django.core.exceptions import ValidationError as DjangoValidationError
from rest_framework import serializers

from .models import User


EMAIL_RE = re.compile(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
PHONE_RE = re.compile(r'^07\d{8}$')
PASSWORD_RULES = (
    (re.compile(r'[A-Z]'), 'Password must contain at least one uppercase letter.'),
    (re.compile(r'[a-z]'), 'Password must contain at least one lowercase letter.'),
    (re.compile(r'\d'), 'Password must contain at least one number.'),
    (re.compile(r'[^A-Za-z0-9]'), 'Password must contain at least one special character.'),
)


def normalize_phone_number(value):
    return re.sub(r'[\s-]', '', value.strip())


def validate_strong_password(value, user=None):
    errors = []

    if len(value) < 8:
        errors.append('Password must be at least 8 characters long.')

    for pattern, message in PASSWORD_RULES:
        if not pattern.search(value):
            errors.append(message)

    try:
        django_validate_password(value, user=user)
    except DjangoValidationError as exc:
        errors.extend(exc.messages)

    if errors:
        errors = list(dict.fromkeys(errors))
        raise serializers.ValidationError(errors)


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'full_name', 'email', 'phone_number', 'is_admin', 'created_at')


class RegisterSerializer(serializers.ModelSerializer):
    full_name = serializers.CharField(max_length=255)
    email = serializers.EmailField()
    phone_number = serializers.CharField(max_length=20)
    password = serializers.CharField(write_only=True, trim_whitespace=False)

    class Meta:
        model = User
        fields = ('full_name', 'email', 'phone_number', 'password')

    def validate_full_name(self, value):
        value = ' '.join(value.strip().split())
        if len(value) < 2:
            raise serializers.ValidationError('Full name must be at least 2 characters long.')
        return value

    def validate_email(self, value):
        value = value.strip().lower()
        if not EMAIL_RE.match(value):
            raise serializers.ValidationError('Enter a valid email address, for example user@example.com.')
        if User.objects.filter(email__iexact=value).exists():
            raise serializers.ValidationError('A user with this email already exists.')
        return value

    def validate_phone_number(self, value):
        value = normalize_phone_number(value)
        if not PHONE_RE.match(value):
            raise serializers.ValidationError('Phone number must be 10 digits and start with 07.')
        return value

    def validate_password(self, value):
        validate_strong_password(value)
        return value

    def validate(self, attrs):
        password_lower = attrs['password'].lower()
        email_local = attrs['email'].split('@', 1)[0].lower()
        name_parts = [part.lower() for part in attrs['full_name'].split() if len(part) >= 3]
        blocked_parts = [email_local, *name_parts]

        if any(part and part in password_lower for part in blocked_parts):
            raise serializers.ValidationError({
                'password': ['Password cannot contain your name or email.']
            })

        return attrs

    def create(self, validated_data):
        user = User.objects.create_user(
            email=validated_data['email'],
            password=validated_data['password'],
            full_name=validated_data['full_name'],
            phone_number=validated_data.get('phone_number', '')
        )
        return user


class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField()

    def validate(self, attrs):
        email = attrs.get('email')
        password = attrs.get('password')

        if email and password:
            user = authenticate(request=self.context.get('request'), email=email, password=password)
            if not user:
                raise serializers.ValidationError('Invalid email or password', code='authorization')
        else:
            raise serializers.ValidationError('Must include "email" and "password".', code='authorization')

        attrs['user'] = user
        return attrs


class ChangePasswordSerializer(serializers.Serializer):
    old_password = serializers.CharField(required=True)
    new_password = serializers.CharField(required=True, trim_whitespace=False)

    def validate_old_password(self, value):
        user = self.context['request'].user
        if not user.check_password(value):
            raise serializers.ValidationError("Old password is not correct")
        return value

    def validate_new_password(self, value):
        user = self.context['request'].user
        validate_strong_password(value, user=user)
        return value
