"""
Contract testing between microservices.

Defines and validates contracts between different services to ensure
compatibility and prevent breaking changes during development.
"""

import pytest
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from unittest.mock import Mock, AsyncMock
from datetime import datetime
import jsonschema
from jsonschema import validate, ValidationError as JsonSchemaValidationError

from music_gen.core.interfaces.services import (
    GenerationRequest,
    GenerationResult,
    TrainingConfig,
)


@dataclass
class ServiceContract:
    """Represents a contract between services."""

    service_name: str
    version: str
    provider: str
    consumer: str
    interactions: List[Dict[str, Any]]
    schemas: Dict[str, Dict[str, Any]]


class ContractValidator:
    """Validates service contracts and schemas."""

    def __init__(self):
        self.contracts: Dict[str, ServiceContract] = {}
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self._load_base_schemas()

    def _load_base_schemas(self):
        """Load base schemas for core data types."""
        # Generation Request Schema
        self.schemas["GenerationRequest"] = {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "minLength": 1, "maxLength": 1000},
                "duration": {"type": "number", "minimum": 0.1, "maximum": 300.0},
                "model_id": {"type": "string", "pattern": "^[a-zA-Z0-9/_-]+$"},
                "temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 1000},
                "top_p": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "seed": {"type": "integer", "minimum": 0},
            },
            "required": ["prompt"],
            "additionalProperties": False,
        }

        # Generation Result Schema
        self.schemas["GenerationResult"] = {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "pattern": "^[a-zA-Z0-9_-]+$"},
                "audio_url": {"type": "string", "format": "uri"},
                "duration": {"type": "number", "minimum": 0.0},
                "sample_rate": {"type": "integer", "minimum": 8000, "maximum": 192000},
                "status": {
                    "type": "string",
                    "enum": ["pending", "processing", "completed", "failed", "cancelled"],
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "model_id": {"type": "string"},
                        "generation_time": {
                            "type": "object",
                            "properties": {"total_seconds": {"type": "number", "minimum": 0}},
                        },
                    },
                },
            },
            "required": ["task_id", "status"],
            "additionalProperties": True,
        }

        # Task Status Schema
        self.schemas["TaskStatus"] = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": ["pending", "processing", "completed", "failed", "cancelled"],
                },
                "progress": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "created_at": {"type": "string", "format": "date-time"},
                "updated_at": {"type": "string", "format": "date-time"},
                "error": {"type": "string"},
                "result": {"$ref": "#/definitions/GenerationResult"},
            },
            "required": ["id", "status"],
            "additionalProperties": False,
        }

        # Model Info Schema
        self.schemas["ModelInfo"] = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
                "type": {"type": "string", "enum": ["pretrained", "custom", "fine_tuned"]},
                "size_mb": {"type": "number", "minimum": 0},
                "description": {"type": "string"},
                "capabilities": {
                    "type": "object",
                    "properties": {
                        "max_duration": {"type": "number"},
                        "sample_rates": {"type": "array", "items": {"type": "integer"}},
                        "conditioning": {"type": "boolean"},
                    },
                },
            },
            "required": ["id", "name", "type"],
            "additionalProperties": False,
        }

    def register_contract(self, contract: ServiceContract):
        """Register a service contract."""
        contract_key = f"{contract.provider}-{contract.consumer}"
        self.contracts[contract_key] = contract

    def validate_request(self, schema_name: str, data: Dict[str, Any]) -> bool:
        """Validate request data against schema."""
        try:
            schema = self.schemas.get(schema_name)
            if not schema:
                raise ValueError(f"Schema {schema_name} not found")

            validate(instance=data, schema=schema)
            return True
        except JsonSchemaValidationError as e:
            raise ValueError(f"Validation failed: {e.message}")

    def validate_response(self, schema_name: str, data: Dict[str, Any]) -> bool:
        """Validate response data against schema."""
        return self.validate_request(schema_name, data)

    def check_contract_compatibility(
        self, provider_contract: ServiceContract, consumer_contract: ServiceContract
    ) -> Dict[str, Any]:
        """Check compatibility between provider and consumer contracts."""
        compatibility_report = {"compatible": True, "issues": [], "warnings": []}

        # Check version compatibility
        if provider_contract.version != consumer_contract.version:
            compatibility_report["warnings"].append(
                f"Version mismatch: provider {provider_contract.version} vs consumer {consumer_contract.version}"
            )

        # Check interaction compatibility
        provider_interactions = {i["name"]: i for i in provider_contract.interactions}
        consumer_interactions = {i["name"]: i for i in consumer_contract.interactions}

        for interaction_name, consumer_interaction in consumer_interactions.items():
            if interaction_name not in provider_interactions:
                compatibility_report["issues"].append(
                    f"Consumer expects interaction '{interaction_name}' not provided by provider"
                )
                compatibility_report["compatible"] = False
            else:
                provider_interaction = provider_interactions[interaction_name]

                # Check request/response schema compatibility
                self._check_schema_compatibility(
                    provider_interaction.get("request_schema"),
                    consumer_interaction.get("request_schema"),
                    f"{interaction_name} request",
                    compatibility_report,
                )

                self._check_schema_compatibility(
                    provider_interaction.get("response_schema"),
                    consumer_interaction.get("response_schema"),
                    f"{interaction_name} response",
                    compatibility_report,
                )

        return compatibility_report

    def _check_schema_compatibility(
        self,
        provider_schema: Optional[Dict],
        consumer_schema: Optional[Dict],
        context: str,
        compatibility_report: Dict[str, Any],
    ):
        """Check compatibility between provider and consumer schemas."""
        if not provider_schema or not consumer_schema:
            return

        # Simple compatibility check - in practice, this would be more sophisticated
        provider_required = set(provider_schema.get("required", []))
        consumer_required = set(consumer_schema.get("required", []))

        # Consumer requires more fields than provider provides
        missing_required = consumer_required - provider_required
        if missing_required:
            compatibility_report["issues"].append(
                f"{context}: Consumer requires fields {missing_required} not guaranteed by provider"
            )
            compatibility_report["compatible"] = False

        # Check property types
        provider_props = provider_schema.get("properties", {})
        consumer_props = consumer_schema.get("properties", {})

        for prop_name, consumer_prop in consumer_props.items():
            if prop_name in provider_props:
                provider_prop = provider_props[prop_name]
                if provider_prop.get("type") != consumer_prop.get("type"):
                    compatibility_report["warnings"].append(
                        f"{context}: Property '{prop_name}' type mismatch"
                    )


@pytest.mark.contract
class TestGenerationServiceContract:
    """Test contracts for the generation service."""

    @pytest.fixture
    def contract_validator(self):
        """Create contract validator."""
        return ContractValidator()

    @pytest.fixture
    def generation_service_contract(self):
        """Define generation service contract."""
        return ServiceContract(
            service_name="GenerationService",
            version="1.0",
            provider="generation_service",
            consumer="api_gateway",
            interactions=[
                {
                    "name": "generate_music",
                    "type": "request_response",
                    "request_schema": "GenerationRequest",
                    "response_schema": "GenerationResult",
                    "description": "Generate music from text prompt",
                },
                {
                    "name": "get_generation_status",
                    "type": "request_response",
                    "request_schema": {
                        "type": "object",
                        "properties": {"task_id": {"type": "string"}},
                        "required": ["task_id"],
                    },
                    "response_schema": "TaskStatus",
                },
                {
                    "name": "cancel_generation",
                    "type": "request_response",
                    "request_schema": {
                        "type": "object",
                        "properties": {"task_id": {"type": "string"}},
                        "required": ["task_id"],
                    },
                    "response_schema": {
                        "type": "object",
                        "properties": {
                            "success": {"type": "boolean"},
                            "message": {"type": "string"},
                        },
                        "required": ["success"],
                    },
                },
            ],
            schemas={},
        )

    def test_generation_request_contract(self, contract_validator):
        """Test generation request contract validation."""
        # Valid request
        valid_request = {
            "prompt": "Upbeat jazz music with piano",
            "duration": 15.0,
            "temperature": 0.8,
            "model_id": "facebook/musicgen-small",
        }

        assert contract_validator.validate_request("GenerationRequest", valid_request)

        # Invalid requests
        invalid_requests = [
            {},  # Missing prompt
            {"prompt": ""},  # Empty prompt
            {"prompt": "test", "duration": -1},  # Invalid duration
            {"prompt": "test", "temperature": 3.0},  # Invalid temperature
            {"prompt": "A" * 1001},  # Prompt too long
        ]

        for invalid_request in invalid_requests:
            with pytest.raises(ValueError):
                contract_validator.validate_request("GenerationRequest", invalid_request)

    def test_generation_result_contract(self, contract_validator):
        """Test generation result contract validation."""
        # Valid result
        valid_result = {
            "task_id": "task_123",
            "status": "completed",
            "audio_url": "https://example.com/audio.wav",
            "duration": 15.0,
            "sample_rate": 24000,
            "metadata": {
                "prompt": "Test music",
                "model_id": "facebook/musicgen-small",
                "generation_time": {"total_seconds": 8.5},
            },
        }

        assert contract_validator.validate_response("GenerationResult", valid_result)

        # Invalid results
        invalid_results = [
            {},  # Missing required fields
            {"task_id": "test"},  # Missing status
            {"task_id": "test", "status": "invalid_status"},  # Invalid status
            {"task_id": "test", "status": "completed", "sample_rate": 1000},  # Invalid sample rate
        ]

        for invalid_result in invalid_results:
            with pytest.raises(ValueError):
                contract_validator.validate_response("GenerationResult", invalid_result)

    def test_task_status_contract(self, contract_validator):
        """Test task status contract validation."""
        # Valid status
        valid_status = {
            "id": "task_456",
            "status": "processing",
            "progress": 0.65,
            "created_at": "2024-01-01T12:00:00Z",
            "updated_at": "2024-01-01T12:05:00Z",
        }

        assert contract_validator.validate_response("TaskStatus", valid_status)

        # Invalid status objects
        invalid_statuses = [
            {},  # Missing required fields
            {"id": "test", "status": "unknown"},  # Invalid status value
            {"id": "test", "status": "processing", "progress": 1.5},  # Invalid progress
        ]

        for invalid_status in invalid_statuses:
            with pytest.raises(ValueError):
                contract_validator.validate_response("TaskStatus", invalid_status)


@pytest.mark.contract
class TestModelServiceContract:
    """Test contracts for the model service."""

    @pytest.fixture
    def model_service_contract(self):
        """Define model service contract."""
        return ServiceContract(
            service_name="ModelService",
            version="1.0",
            provider="model_service",
            consumer="generation_service",
            interactions=[
                {
                    "name": "list_models",
                    "type": "request_response",
                    "request_schema": {
                        "type": "object",
                        "properties": {
                            "type_filter": {
                                "type": "string",
                                "enum": ["pretrained", "custom", "fine_tuned"],
                            }
                        },
                        "additionalProperties": False,
                    },
                    "response_schema": {
                        "type": "object",
                        "properties": {
                            "models": {
                                "type": "array",
                                "items": {"$ref": "#/definitions/ModelInfo"},
                            }
                        },
                        "required": ["models"],
                    },
                },
                {
                    "name": "load_model",
                    "type": "request_response",
                    "request_schema": {
                        "type": "object",
                        "properties": {
                            "model_id": {"type": "string"},
                            "device": {"type": "string", "enum": ["cpu", "cuda", "mps"]},
                        },
                        "required": ["model_id"],
                    },
                    "response_schema": {
                        "type": "object",
                        "properties": {
                            "success": {"type": "boolean"},
                            "model_info": {"$ref": "#/definitions/ModelInfo"},
                            "load_time_seconds": {"type": "number", "minimum": 0},
                        },
                        "required": ["success"],
                    },
                },
            ],
            schemas={},
        )

    def test_model_info_contract(self, contract_validator):
        """Test model info contract validation."""
        # Valid model info
        valid_model = {
            "id": "facebook/musicgen-small",
            "name": "MusicGen Small",
            "type": "pretrained",
            "size_mb": 300.5,
            "description": "Small model for quick generation",
            "capabilities": {
                "max_duration": 60.0,
                "sample_rates": [24000, 32000],
                "conditioning": True,
            },
        }

        assert contract_validator.validate_response("ModelInfo", valid_model)

        # Invalid model info
        invalid_models = [
            {},  # Missing required fields
            {"id": "test", "name": "Test", "type": "invalid_type"},  # Invalid type
            {"id": "test", "name": "Test", "type": "pretrained", "size_mb": -1},  # Invalid size
        ]

        for invalid_model in invalid_models:
            with pytest.raises(ValueError):
                contract_validator.validate_response("ModelInfo", invalid_model)


@pytest.mark.contract
class TestServiceContractCompatibility:
    """Test compatibility between service contracts."""

    def test_contract_compatibility_check(self):
        """Test contract compatibility checking."""
        validator = ContractValidator()

        # Define provider contract
        provider_contract = ServiceContract(
            service_name="MusicAPI",
            version="1.0",
            provider="music_service",
            consumer="api_gateway",
            interactions=[
                {
                    "name": "generate",
                    "request_schema": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"},
                            "duration": {"type": "number"},
                        },
                        "required": ["prompt"],
                    },
                    "response_schema": {
                        "type": "object",
                        "properties": {"task_id": {"type": "string"}, "status": {"type": "string"}},
                        "required": ["task_id", "status"],
                    },
                }
            ],
            schemas={},
        )

        # Define compatible consumer contract
        compatible_consumer = ServiceContract(
            service_name="MusicAPI",
            version="1.0",
            provider="music_service",
            consumer="web_client",
            interactions=[
                {
                    "name": "generate",
                    "request_schema": {
                        "type": "object",
                        "properties": {"prompt": {"type": "string"}},
                        "required": ["prompt"],
                    },
                    "response_schema": {
                        "type": "object",
                        "properties": {"task_id": {"type": "string"}, "status": {"type": "string"}},
                        "required": ["task_id"],
                    },
                }
            ],
            schemas={},
        )

        # Check compatibility
        compatibility = validator.check_contract_compatibility(
            provider_contract, compatible_consumer
        )

        assert compatibility["compatible"] is True
        assert len(compatibility["issues"]) == 0

    def test_contract_incompatibility_detection(self):
        """Test detection of contract incompatibilities."""
        validator = ContractValidator()

        # Define provider contract
        provider_contract = ServiceContract(
            service_name="MusicAPI",
            version="1.0",
            provider="music_service",
            consumer="api_gateway",
            interactions=[
                {
                    "name": "generate",
                    "request_schema": {
                        "type": "object",
                        "properties": {"prompt": {"type": "string"}},
                        "required": ["prompt"],
                    },
                    "response_schema": {
                        "type": "object",
                        "properties": {"task_id": {"type": "string"}},
                        "required": ["task_id"],
                    },
                }
            ],
            schemas={},
        )

        # Define incompatible consumer contract
        incompatible_consumer = ServiceContract(
            service_name="MusicAPI",
            version="2.0",  # Different version
            provider="music_service",
            consumer="mobile_client",
            interactions=[
                {
                    "name": "generate",
                    "request_schema": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"},
                            "user_id": {"type": "string"},  # Additional required field
                        },
                        "required": ["prompt", "user_id"],
                    },
                    "response_schema": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "string"},
                            "status": {"type": "string"},
                            "metadata": {"type": "object"},  # Additional required field
                        },
                        "required": ["task_id", "status", "metadata"],
                    },
                },
                {
                    "name": "get_user_history",  # Additional interaction
                    "request_schema": {
                        "type": "object",
                        "properties": {"user_id": {"type": "string"}},
                        "required": ["user_id"],
                    },
                },
            ],
            schemas={},
        )

        # Check compatibility
        compatibility = validator.check_contract_compatibility(
            provider_contract, incompatible_consumer
        )

        assert compatibility["compatible"] is False
        assert len(compatibility["issues"]) > 0
        assert len(compatibility["warnings"]) > 0

        # Should detect missing interaction
        missing_interaction_issue = any(
            "get_user_history" in issue for issue in compatibility["issues"]
        )
        assert missing_interaction_issue

        # Should detect version mismatch
        version_warning = any(
            "Version mismatch" in warning for warning in compatibility["warnings"]
        )
        assert version_warning


@pytest.mark.contract
class TestContractEvolution:
    """Test contract evolution and backward compatibility."""

    def test_backward_compatible_evolution(self):
        """Test backward compatible contract evolution."""
        validator = ContractValidator()

        # Original contract (v1.0)
        original_contract = ServiceContract(
            service_name="GenerationAPI",
            version="1.0",
            provider="generation_service",
            consumer="api_gateway",
            interactions=[
                {
                    "name": "generate",
                    "request_schema": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"},
                            "duration": {"type": "number"},
                        },
                        "required": ["prompt"],
                    },
                    "response_schema": {
                        "type": "object",
                        "properties": {"task_id": {"type": "string"}, "status": {"type": "string"}},
                        "required": ["task_id", "status"],
                    },
                }
            ],
            schemas={},
        )

        # Evolved contract (v1.1) - backward compatible
        evolved_contract = ServiceContract(
            service_name="GenerationAPI",
            version="1.1",
            provider="generation_service",
            consumer="api_gateway",
            interactions=[
                {
                    "name": "generate",
                    "request_schema": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"},
                            "duration": {"type": "number"},
                            "style": {"type": "string"},  # New optional field
                        },
                        "required": ["prompt"],  # Same required fields
                    },
                    "response_schema": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "string"},
                            "status": {"type": "string"},
                            "estimated_duration": {"type": "number"},  # New optional field
                        },
                        "required": ["task_id", "status"],  # Same required fields
                    },
                }
            ],
            schemas={},
        )

        # Check compatibility (old consumer with new provider)
        compatibility = validator.check_contract_compatibility(evolved_contract, original_contract)

        # Should be compatible (new provider can serve old consumer)
        assert compatibility["compatible"] is True

    def test_breaking_change_detection(self):
        """Test detection of breaking changes."""
        validator = ContractValidator()

        # Original contract
        original_contract = ServiceContract(
            service_name="GenerationAPI",
            version="1.0",
            provider="generation_service",
            consumer="api_gateway",
            interactions=[
                {
                    "name": "generate",
                    "request_schema": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"},
                            "duration": {"type": "number"},
                        },
                        "required": ["prompt"],
                    },
                    "response_schema": {
                        "type": "object",
                        "properties": {"task_id": {"type": "string"}, "status": {"type": "string"}},
                        "required": ["task_id", "status"],
                    },
                }
            ],
            schemas={},
        )

        # Breaking change contract (v2.0)
        breaking_contract = ServiceContract(
            service_name="GenerationAPI",
            version="2.0",
            provider="generation_service",
            consumer="api_gateway",
            interactions=[
                {
                    "name": "generate",
                    "request_schema": {
                        "type": "object",
                        "properties": {
                            "text_prompt": {"type": "string"},  # Renamed field
                            "duration": {"type": "number"},
                            "api_key": {"type": "string"},  # New required field
                        },
                        "required": ["text_prompt", "api_key"],  # Changed requirements
                    },
                    "response_schema": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},  # Renamed field
                            "state": {"type": "string"},  # Renamed field
                        },
                        "required": ["id", "state"],
                    },
                }
            ],
            schemas={},
        )

        # Check compatibility (old consumer with new provider)
        compatibility = validator.check_contract_compatibility(breaking_contract, original_contract)

        # Should detect breaking changes
        assert compatibility["compatible"] is False
        assert len(compatibility["issues"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
