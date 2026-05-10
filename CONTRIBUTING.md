# 贡献指南 / Contributing Guidelines

*(English version follows the Chinese section)*

感谢你对 EigenTruth 的关注！我们欢迎所有形式的贡献。

## 开发环境设置

```bash
git clone https://github.com/catamitez0-maker/EigenTruth.git
cd EigenTruth
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .[dev]
```

## 开发流程

1. Fork 本仓库
2. 创建特性分支: `git checkout -b feature/my-feature`
3. 编写代码和测试
4. 运行测试: `pytest tests/ -v`
5. 代码检查: `ruff check src/`
6. 提交 PR

## 代码规范

- 使用 `ruff` 进行代码格式和风格检查
- 所有公开函数必须有 docstring (文档字符串)
- 新功能必须附带单元测试
- 数值计算必须遵循 FP32 安全规范与批处理(Batching)安全（参见 `core/math_engine.py`）

## 提交信息格式

```
<type>: <description>

[optional body]
```

类型包括: `feat`, `fix`, `docs`, `test`, `refactor`, `ci`

<br>
<hr>
<br>

Thank you for your interest in EigenTruth! We welcome all forms of contribution.

## Development Environment Setup

```bash
git clone https://github.com/catamitez0-maker/EigenTruth.git
cd EigenTruth
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .[dev]
```

## Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Write your code and tests
4. Run tests: `pytest tests/ -v`
5. Run linting: `ruff check src/`
6. Submit a Pull Request (PR)

## Code Standards

- Use `ruff` for code formatting and style checking
- All public functions must have docstrings
- New features must be accompanied by unit tests
- Numerical computations must adhere to FP32 and Batching safety standards (see `core/math_engine.py`)

## Commit Message Format

```
<type>: <description>

[optional body]
```

Allowed types include: `feat`, `fix`, `docs`, `test`, `refactor`, `ci`
