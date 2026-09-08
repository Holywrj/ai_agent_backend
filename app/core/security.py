from passlib.context import CryptContext

# Passlib 的“密码算法管理器”
# schemes: 密码 Hash 算法, 支持多个hash算法共存，schemes=["bcrypt", "argon2"]
# deprecated: 配置了多个密码 Hash 算法，旧算法可以被标记为过时，并在用户登录成功后逐步升级
# deprecated='auto', 让 Passlib 根据 schemes 的配置自动判断哪些是 deprecated
pwd_context = CryptContext(
    schemes=['bcrypt'],
    deprecated='auto'
)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    return pwd_context.verify(password, hashed_password)
