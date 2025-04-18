# Changelog

All notable changes to this project will be documented in this file.

## [0.1.14] - 2025-04-18

### Added
- DeepSeek model integration with full support for tool execution
- New Orchestrator Agent Mode:
  - Advanced planning for complex multi-step tasks
  - Strategic delegation across multiple MCP servers
  - Intelligent agent coordination and communication
  - Parallel task execution capabilities
  - Dynamic resource allocation
  - Sophisticated workflow management
  - Real-time progress monitoring
  - Adaptive task prioritization
- Client-Side Sampling Support:
  - Dynamic sampling configuration from client
  - Flexible LLM response generation
  - Customizable sampling parameters
  - Real-time sampling adjustments
- Chat History File Storage:
  - Save complete chat conversations to files
  - Load previous conversations from saved files
  - Continue conversations from where you left off
  - File-based backup and restoration
  - Persistent chat history across sessions

### Changed
- Enhanced Mode System with three distinct modes:
  - Chat Mode (Default)
  - Autonomous Mode
  - Orchestrator Mode
- Updated AI model integration documentation
- Improved chat history management system
- Enhanced server configuration options for new features

### Fixed
- Improved mode switching reliability
- Enhanced chat history persistence
- Optimized orchestrator mode performance

## [0.1.13] - 2025-04-14

### Added
- Gemini model integration with full support for tool execution
- Redis-powered memory persistence:
  - Conversation history tracking
  - State management across sessions
  - Configurable memory retention
  - Efficient data serialization and retrieval
  - Multi-server memory synchronization
- Agentic Mode capabilities:
  - Autonomous task execution without human intervention
  - Advanced reasoning and decision-making
  - Complex task decomposition and handling
  - Self-guided tool selection and execution
- Advanced prompt features:
  - Dynamic prompt discovery across servers
  - JSON and key-value format support
  - Nested argument structures
  - Automatic type conversion and validation
- Comprehensive troubleshooting guide with:
  - Common issues and solutions
  - Debug mode instructions
  - Support workflow
- Detailed architecture documentation with component breakdown
- Advanced server configuration examples for:
  - Multiple transport protocols
  - Various LLM providers
  - Docker integration

### Changed
- Enhanced installation process with UV package manager
- Improved development quick start guide
- Updated server configuration format to support multiple LLM providers
- Expanded model support documentation for all providers
- Enhanced security documentation with explicit user control details
- Restructured README with clearer sections and examples

### Fixed
- Standardized command formatting in documentation
- Improved code block consistency
- Enhanced example clarity and completeness

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