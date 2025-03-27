# Changelog

All notable changes to this project will be documented in this file.

## [0.1.1] - 2025-03-27

### Added
- Comprehensive Security & Privacy section with detailed subsections:
  - Explicit User Control
  - Data Protection
  - Privacy-First Approach
  - Secure Communication
- Detailed Model Support section covering:
  - OpenAI Models
  - OpenRouter Models
  - Groq Models
  - Universal Model Support through ReAct Agent
- Structured Testing section with:
  - Multiple test running options
  - Test directory structure
  - Coverage reporting instructions
- Support for additional LLM providers:
  - OpenRouter integration
  - Groq integration
  - Universal model support through ReAct Agent

### Changed
- Improved AI-Powered Intelligence section:
  - Added support for multiple LLM providers (OpenAI, OpenRouter, Groq)
  - Added detailed ReAct Agent capabilities for models without function calling
  - Fixed typos in "seamless"
- Enhanced Server Configuration Examples:
  - Added support for multiple LLM providers
  - Updated model examples
  - Added comments for supported providers
- Updated Prerequisites:
  - Changed Python version requirement from 3.12+ to 3.10+
  - Updated API key requirements to support multiple providers
- Improved environment variable setup:
  - Changed from OPENAI_API_KEY to LLM_API_KEY for broader provider support
  - Added support for multiple API keys in .env file

### Fixed
- Typos in model integration descriptions
- Formatting issues in various sections
- Inconsistent capitalization in headers
- Fixed typo in "client" command (was "cient")
- Improved code block formatting and consistency

### Removed
- Redundant security information
- Simplified test section
- Removed specific OpenAI model references in favor of provider-agnostic examples
- Removed redundant prompt examples in favor of more structured documentation

## [0.1.0] - 2025-03-21
- Initial release
- Basic MCP server integration
- OpenAI model support
- Core CLI functionality 