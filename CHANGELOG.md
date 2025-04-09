# Changelog

## [Unreleased]
### Changed
- Removed BigQuery integration and all GCP dependencies
- Updated model from mixtral-8x7b-32768 to llama3-70b-8192
- Added handle_parsing_errors=True to AgentExecutor
- Improved error handling and initialization
- Enhanced initial response handling

### Fixed
- Resolved environment variable loading issues
- Fixed output parsing errors
- Improved tool selection logic
- Maintained core functionality through all changes

### Known Issues
- LangSmith API key warning (non-critical)
- Initial response could be more natural
