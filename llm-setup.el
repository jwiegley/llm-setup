;;; llm-setup.el --- LLM model management  -*- lexical-binding: t; -*-

;; Copyright (C) 2026 John Wiegley

;; Author: John Wiegley <johnw@newartisans.com>
;; Version: 0.1.0
;; Package-Requires: ((emacs "27.1"))
;; Keywords: tools, processes
;; URL: https://github.com/jwiegley/dot-emacs

;;; Commentary:

;; Model management utility for GGUF, MLX, LMStudio, and Ollama models.
;; Handles downloading, importing, configuration, and serving of AI models.
;; Supports llama-cpp, mlx-lm, and vllm-mlx serving engines.

;;; Code:

(require 'cl-lib)
(require 'json)
(require 'url)
(require 'files)
(require 'subr-x)

(declare-function yaml-mode "yaml-mode" ())
(declare-function json-mode "json-mode" ())
(declare-function lookup-password "lookup-password")
(declare-function gptel-backends-make-litellm "gptel-ext")

(defvar gptel-model)
(defvar gptel-backend)

(defgroup llm-setup nil
  "Model management configuration."
  :group 'tools)

(defcustom llm-setup-protocol "https"
  "Protocol for model server."
  :type 'string
  :group 'llm-setup)

(defcustom llm-setup-server "hera.lan"
  "Server address."
  :type 'string
  :group 'llm-setup)

(defcustom llm-setup-port 8443
  "Server port."
  :type 'integer
  :group 'llm-setup)

(defcustom llm-setup-prefix ""
  "API prefix."
  :type 'string
  :group 'llm-setup)

(defcustom llm-setup-api-key "sk-1234"
  "API key."
  :type 'string
  :group 'llm-setup)

(defcustom llm-setup-threads 24
  "Number of threads."
  :type 'integer
  :group 'llm-setup)

(defcustom llm-setup-default-hostname "hera"
  "Name of model host."
  :type 'string
  :group 'llm-setup)

(defcustom llm-setup-default-instance-name 'Qwen3.5-27B
  "Name of default instance."
  :type 'symbol
  :group 'llm-setup)

(defcustom llm-setup-valid-hostnames '("hera" "clio" ;; "vulcan"
    )
  "Name of hosts that can run models."
  :type '(repeat string)
  :group 'llm-setup)

(defcustom llm-setup-llama-server-executable "llama-server"
  "Path to the llama-server executable."
  :type 'file
  :group 'llm-setup)

(defcustom llm-setup-mlx-lm-executable "mlx-lm"
  "Path to the mlx-lm executable."
  :type 'file
  :group 'llm-setup)

(defcustom llm-setup-vllm-mlx-executable "vllm-mlx"
  "Path to the vllm-mlx executable."
  :type 'file
  :group 'llm-setup)

(defcustom llm-setup-llama-swap-prolog "
healthCheckTimeout: 7200
startPort: 9400
"
  "Prolog for beginning of llama-swap.yaml file."
  :type 'string
  :group 'llm-setup)

(defcustom llm-setup-llama-swap-epilog "
groups:
  always_on:
    swap: false
    exclusive: false
    members:
      - Devstral-2-123B-Instruct-2512
      - Devstral-Small-2-24B-Instruct-2512
      - Qwen3-Coder-Next
      - gpt-oss-20b
      - mlx-community/gpt-oss-20b-MXFP4-Q8
      - gpt-oss-safeguard-20b
      - lmstudio-community/gpt-oss-safeguard-20b-MLX-MXFP4
      - gpt-oss-120b
      - Qwen3.5-27B
      - Qwen3.5-27B-Instruct
      - Qwen3.5-9B
      - Qwen3.5-9B-Instruct
      - Qwen3.5-4B
      - Qwen3.5-4B-Instruct
      - Qwen3.5-2B
      - Qwen3.5-2B-Instruct
      - Qwen3.5-0.8B
      - Qwen3.5-35B-A3B
      - Qwen3.5-397B-A17B
      - bge-m3

  # Only one of these can be loaded at a time
  large_models:
    swap: true
    exclusive: false
    members:
      - GLM-5.1
      - Kimi-K2.5
      - Llama-4-Maverick-17B-128E-Instruct
      - Llama-4-Scout-17B-16E-Instruct
      - Phi-4-reasoning-plus
      - mlx-community/gpt-oss-120b-MXFP4-Q8

  # Only one of these can be loaded at a time
  embeddings:
    swap: true
    exclusive: false
    members:
      - NV-Embed-v2
      - Qwen3-Embedding-8B
      - all-MiniLM-L6-v2
      - bge-base-en-v1.5
      - bge-large-en-v1.5
      - nomic-embed-text-v2-moe

  # Only one of these can be loaded at a time
  rerankings:
    swap: true
    exclusive: false
    members:
      - Qwen.Qwen3-Reranker-8B
      - bge-reranker-v2-m3

  # Only one of these can be loaded at a time
  stt:
    swap: true
    exclusive: false
    members:
      - whisper-large-v3-mlx
"
  "Epilog for beginning of llama-swap.yaml file."
  :type 'string
  :group 'llm-setup)

(defcustom llm-setup-litellm-path
  "/ssh:vulcan|sudo:root@vulcan:/etc/litellm/config.yaml"
  "Pathname to LiteLLM's config.yaml file."
  :type 'file
  :group 'llm-setup)

(defcustom llm-setup-litellm-prolog ""
  "Prolog for beginning of LiteLLM's config.yaml file."
  :type 'string
  :group 'llm-setup)

(defcustom llm-setup-litellm-environment-function
  (lambda ()
    (format-spec
     "
environment_variables:
  ANTHROPIC_API_KEY: \"%a\"
  GEMINI_API_KEY: \"%g\"
  OPENAI_API_KEY: \"%o\"
  PERPLEXITYAI_API_KEY: \"%p\"
  GROQ_API_KEY: \"%r\"
  OPENROUTER_API_KEY: \"%e\"
  POSITRON_ANTHROPIC_API_KEY: \"%A\"
  POSITRON_GEMINI_API_KEY: \"%G\"
  POSITRON_OPENAI_API_KEY: \"%O\"
  POSITRON_API_KEY: \"%P\"
"
     `((?a
        .
        ,(lambda ()
           (lookup-password "api.anthropic.com" "johnw" 443)))
       (?g
        . ,(lambda () (lookup-password "api.gemini.com" "johnw" 443)))
       (?o
        . ,(lambda () (lookup-password "api.openai.com" "johnw" 443)))
       (?p
        .
        ,(lambda ()
           (lookup-password "api.perplexity.ai" "johnw" 443)))
       (?r
        . ,(lambda () (lookup-password "api.groq.com" "johnw" 443)))
       (?e
        . ,(lambda () (lookup-password "openrouter.ai" "johnw" 443)))
       (?A
        .
        ,(lambda ()
           (lookup-password
            "positron@api.anthropic.com" "jwiegley" 443)))
       (?G
        .
        ,(lambda ()
           (lookup-password
            "positron@api.gemini.com" "jwiegley" 443)))
       (?O
        .
        ,(lambda ()
           (lookup-password
            "positron@api.openai.com" "jwiegley" 443)))
       (?P
        .
        ,(lambda ()
           (lookup-password
            "positron@api-dev.positron.ai" "jwiegley" 443))))))
  "Function for generating credentials for LiteLLM's config.yaml file."
  :type 'function
  :group 'llm-setup)

(defcustom llm-setup-litellm-credentials "
credential_list:
  - credential_name: hera_llama_swap_credential
    credential_values:
      api_base: https://hera.lan:8443/v1
      api_key: \"fake\"
    credential_info:
      description: \"API Key for llama-swap on Hera\"

  - credential_name: hera_vibe_proxy_credential
    credential_values:
      api_base: http://hera.lan:8317/v1
      api_key: \"fake\"
    credential_info:
      description: \"API Key for vibe-proxy on Hera\"

  - credential_name: clio_llama_swap_credential
    credential_values:
      api_base: https://clio.lan:8443/v1
      api_key: \"fake\"
    credential_info:
      description: \"API Key for llama-swap on Clio\"

  - credential_name: openai_credential
    credential_values:
      api_key: os.environ/OPENAI_API_KEY
    credential_info:
      description: \"API Key for OpenAI\"

  - credential_name: anthropic_credential
    credential_values:
      api_key: os.environ/ANTHROPIC_API_KEY
    credential_info:
      description: \"API Key for Anthropic\"

  - credential_name: perplexity_credential
    credential_values:
      api_key: os.environ/PERPLEXITYAI_API_KEY
    credential_info:
      description: \"API Key for Perplexity\"

  - credential_name: groq_credential
    credential_values:
      api_key: os.environ/GROQ_API_KEY
    credential_info:
      description: \"API Key for Groq\"

  - credential_name: openrouter_credential
    credential_values:
      api_key: os.environ/OPENROUTER_API_KEY
    credential_info:
      description: \"API Key for OpenRouter\"

  - credential_name: positron_openai_credential
    credential_values:
      api_key: os.environ/POSITRON_OPENAI_API_KEY
    credential_info:
      description: \"API Key for OpenAI (Positron)\"

  - credential_name: positron_anthropic_credential
    credential_values:
      api_key: os.environ/POSITRON_ANTHROPIC_API_KEY
    credential_info:
      description: \"API Key for Anthropic (Positron)\"

  - credential_name: positron_gemini_credential
    credential_values:
      api_key: os.environ/POSITRON_GEMINI_API_KEY
    credential_info:
      description: \"API Key for Google AI (Positron)\"

  - credential_name: positron_credential
    credential_values:
      api_base: https://api-dev.positron.ai/v1
      api_key: os.environ/POSITRON_API_KEY
    credential_info:
      description: \"API Key for Posistron.ai\"
"
  "Function for generating credentials for LiteLLM's config.yaml file."
  :type 'function
  :group 'llm-setup)

(defcustom llm-setup-litellm-epilog-spec "
litellm_settings:
  request_timeout: 7200
  streaming_request_timeout: 300
  ssl_verify: false
  drop_params: true
  # set_verbose: True
  cache: True
  cache_params:
    type: redis
    host: \"10.0.2.2\"
    port: 8085
    supported_call_types: [\"acompletion\", \"atext_completion\", \"aembedding\", \"atranscription\"]

guardrails:
  - guardrail_name: \"harmony_filter\"
    litellm_params:
      guardrail: harmony_filter.HarmonyResponseFilter
      mode: \"post_call\"
      default_on: true

router_settings:%s
  routing_strategy: \"least-busy\"
  num_retries: 3
  request_timeout: 7200
  streaming_request_timeout: 300
  max_parallel_requests: 100
  allowed_fails: 3
  cooldown_time: 30
  # provider_budget_config:
  #   perplexity:
  #     budget_limit: 5
  #     time_period: 1mo

general_settings:
  store_model_in_db: true
  store_prompts_in_spend_logs: true
  maximum_spend_logs_retention_period: \"90d\"
  maximum_spend_logs_retention_interval: \"7d\"
  enable_pass_through_endpoints: true
"
  "Epilog for LiteLLM's config.yaml file.
Contains a %s placeholder for dynamically generated router fallbacks."
  :type 'string
  :group 'llm-setup)

(defsubst llm-setup-api-base ()
  "Get API base URL."
  (format "%s://%s:%d%s"
          llm-setup-protocol
          llm-setup-server
          llm-setup-port
          llm-setup-prefix))

;; Define paths
(defvar llm-setup-home (expand-file-name "~"))
(defvar llm-setup-xdg-local
  (expand-file-name ".local/share" llm-setup-home))
(defvar llm-setup-gguf-models
  (expand-file-name "Models" llm-setup-home))
(defvar llm-setup-mlx-models
  (expand-file-name ".cache/huggingface/hub" llm-setup-home))
(defvar llm-setup-lmstudio-models
  (expand-file-name "lmstudio/models" llm-setup-xdg-local))
(defvar llm-setup-ollama-models
  (expand-file-name "ollama/models" llm-setup-xdg-local))

(defconst llm-setup-all-model-characteristics
  '(high medium low remote local thinking instruct coding rewrite))

(defconst llm-setup-all-model-capabilities '(media tool json url))

(defconst llm-setup-all-model-mime-types
  '("image/jpeg" "image/png" "image/gif" "image/webp"))

(defconst llm-setup-all-model-kinds
  '(text-generation embedding reranker))

(defconst llm-setup-all-model-providers
  '(local
    vibe-proxy
    openai
    anthropic
    gemini
    positron
    positron_openai
    positron_anthropic
    positron_gemini
    perplexity
    groq
    openrouter))

(defconst llm-setup-all-model-engines
  '(llama-cpp koboldcpp mlx-lm vllm-mlx))

;;; Models have several names:
;;
;; llm-setup-model-name
;; llm-setup-model-aliases
;; llm-setup-instance-name
;; llm-setup-instance-model-name
;;
;; These names are used in several places:
;;
;; - The name of the model as reported by v1/models, which is used for
;;   submitting queries to that model to eihter llama-swap, LiteLLM or
;;   directly to an external model provider (Anthropic, OpenRouter, etc)
;; - The name of the model as managed by that backend
;; - The name of the model as submitted to the provider by the backend
;;
;;; llama-swap.yaml
;;
;; "mlx-community/gpt-oss-20b-MXFP4-Q8":
;;   cmd: >
;;     /etc/profiles/per-user/johnw/bin/mlx-lm server
;;       --host 127.0.0.1 --port ${PORT}
;;       --model mlx-community/gpt-oss-20b-MXFP4-Q8 ...
;;
;; "gpt-oss-20b":
;;   cmd: >
;;     /etc/profiles/per-user/johnw/bin/llama-server
;;       --host 127.0.0.1 --port ${PORT}
;;       --jinja
;;       --model /Users/johnw/Models/unsloth_gpt-oss-20b-GGUF/gpt-oss-20b-F16.gguf ...
;;
;;; litellm/config.yaml

(cl-defstruct
 llm-setup-model
 "Configuration data for a model, and its family of instances."
 name ; name of the model
 description ; description of the model
 characteristics
 (capabilities
  llm-setup-all-model-capabilities) ; capabilities of the model
 (mime-types
  llm-setup-all-model-mime-types) ; MIME types that can be sent
 context-length ; model context length
 max-input-tokens ; number of tokens to accept
 max-output-tokens ; number of tokens to predict
 temperature ; model temperature
 min-p ; minimum p
 top-p ; top p
 top-k ; top k
 (kind 'text-generation) ; nil, or symbol from model-kinds
 (supports-system-message t) ; t if model supports system messages
 (supports-function-calling nil) ; t if model supports function calling
 (supports-reasoning nil) ; t if model supports reasoning
 (supports-response-schema nil) ; t if model supports response schema
 aliases ; model alias names
 instances ; model instances
 )

(cl-defstruct
 llm-setup-instance
 "Configuration data for a model, and its family of instances."
 name ; alternate name to use with provider
 model-name ; alternate model-name to use
 context-length ; context length to use for instance
 max-input-tokens ; number of tokens to accept
 max-output-tokens ; number of tokens to predict
 cache-control ; supports auto-caching?
 (provider 'local) ; where does the model run?
 (parallel 1) ; how many parallel connections to support
 (cache-type-k 'f16) ; K-quantization
 (cache-type-v 'f16) ; V-quantization
 (kv-offload t) ; if nil, emit --no-kv-offload
 (engine 'llama-cpp) ; if local: llama.cpp, koboldcpp, etc.
 (hostnames
  (list llm-setup-default-hostname)) ; if local: hostname where engine runs
 model-path ; if local: path to model directory
 file-path ; if local: (optional) path to model file
 draft-model ; if local: (optional) path to draft model
 arguments ; if local: arguments to engine
 fallbacks ; if remote: list of fallback model names
 (cache-prompt t) ; if nil, emit --no-cache-prompt
 (cache-ram nil) ; if non-nil, emit --cache-ram
 (cache-reuse nil) ; integer: min chunk size for cache reuse
 (slot-save-path nil) ; path for saving/restoring slot KV cache
 (slot-prompt-similarity nil) ; float: min prompt similarity to reuse slot
 )

(defcustom llm-setup-models-list
  (list

   (make-llm-setup-model
    :name 'DeepSeek-R1-0528
    :context-length 163840
    :temperature 0.6
    :min-p 0.01
    :top-p 0.9
    :top-k 20
    :supports-reasoning t
    :instances
    (list
     (make-llm-setup-instance
      :name 'deepseek/deepseek-r1-0528:free
      :provider 'openrouter)))

   (make-llm-setup-model
    :name 'DeepSeek-V3
    :context-length 163840
    :instances
    (list
     (make-llm-setup-instance
      :name 'deepseek-ai/DeepSeek-V3
      :engine 'mlx-lm)))

   (make-llm-setup-model
    :name 'DeepSeek-V3.2
    :context-length 163840
    :instances
    (list
     (make-llm-setup-instance
      :context-length 12000
      :model-path "~/Models/unsloth_DeepSeek-V3.2-GGUF")))

   (make-llm-setup-model
    :name 'Kimi-K2.5
    :context-length 262144
    :temperature 1.0
    :min-p 0.01
    :top-p 0.9
    :top-k 20
    :supports-function-calling t
    :instances
    (list
     (make-llm-setup-instance
      :context-length 98304
      :max-output-tokens 32768
      :model-path "~/Models/unsloth_Kimi-K2.5-GGUF")))

   (make-llm-setup-model
    :name 'SERA-32B
    :context-length 40960
    :temperature 0.7
    :min-p 0.0
    :top-p 0.8
    :top-k 20
    :supports-function-calling t
    :supports-reasoning t
    :instances
    (list
     (make-llm-setup-instance
      :model-path "~/Models/noctrex_SERA-32B-GGUF"
      :hostnames
      '("hera" "clio"))))

   (make-llm-setup-model
    :name 'Llama-4-Scout-17B-16E-Instruct
    :context-length 10485760
    :temperature 0.6
    :min-p 0.01
    :top-p 0.9
    :supports-function-calling t
    :instances
    (list
     (make-llm-setup-instance
      :context-length 1048576
      :model-path "~/Models/unsloth_Llama-4-Scout-17B-16E-Instruct-GGUF"
      :cache-type-k 'q4_1
      :cache-type-v 'q4_1)

     (make-llm-setup-instance
      :name 'meta-llama/llama-4-scout-17b-16e-instruct
      :provider 'groq)))

   ;; (make-llm-setup-model
   ;;  :name 'Llama-4-Maverick-17B-128E-Instruct
   ;;  :context-length 1048576
   ;;  :supports-function-calling t
   ;;  :instances
   ;;  (list
   ;;   (make-llm-setup-instance
   ;;    :context-length 65536
   ;;    :model-path "~/Models/unsloth_Llama-4-Maverick-17B-128E-Instruct-GGUF")

   ;;   (make-llm-setup-instance
   ;;    :name 'meta-llama/llama-4-maverick-17b-128e-instruct
   ;;    :provider 'groq)

   ;;   (make-llm-setup-instance
   ;;    :name 'meta-llama/llama-4-maverick:free
   ;;    :provider 'openrouter)))

   (make-llm-setup-model
    :name 'GLM-4.7-REAP-218B-A32B
    :context-length 202752
    :temperature 0.7
    :min-p 0.01
    :top-p 0.9
    :top-k 40
    :supports-function-calling t
    :supports-reasoning t
    :instances
    (list
     (make-llm-setup-instance
      :model-path "~/Models/unsloth_GLM-4.7-REAP-218B-A32B-GGUF")))

   (make-llm-setup-model
    :name 'GLM-4.7-Flash
    :context-length 202752
    :temperature 0.7
    :min-p 0.01
    :top-p 0.9
    :top-k 40
    :supports-function-calling t
    :supports-reasoning t
    :instances
    (list
     (make-llm-setup-instance
      :model-path "~/Models/unsloth_GLM-4.7-Flash-GGUF"
      :arguments
      '("--repeat-penalty" "1.0"))))

   (make-llm-setup-model
    :name 'GLM-4.7-Flash-REAP-23B-A3B
    :context-length 202752
    :temperature 0.7
    :min-p 0.01
    :top-p 0.9
    :top-k 40
    :supports-function-calling t
    :supports-reasoning t
    :instances
    (list
     (make-llm-setup-instance
      :model-path "~/Models/unsloth_GLM-4.7-Flash-REAP-23B-A3B-GGUF"
      :hostnames
      '("hera" "clio"))))

   (make-llm-setup-model
    :name 'GLM-5
    :context-length 262144
    :temperature 1.0
    :top-p 0.9
    :top-k 40
    :supports-function-calling t
    :supports-reasoning t
    :instances
    (list
     (make-llm-setup-instance
      :model-path "~/Models/unsloth_GLM-5-GGUF")))

   (make-llm-setup-model
    :name 'GLM-5.1
    :context-length 200000
    :temperature 1.0
    :top-p 0.9
    :top-k 40
    :supports-function-calling t
    :supports-reasoning t
    :instances
    (list
     (make-llm-setup-instance
      :model-path "~/Models/unsloth_GLM-5.1-GGUF")))

   (make-llm-setup-model
    :name 'MiniMax-M2-REAP-162B-A10B
    :context-length 262144
    :temperature 1.0
    :top-p 0.9
    :top-k 40
    :supports-function-calling t
    :supports-reasoning t
    :instances
    (list
     (make-llm-setup-instance
      :model-path "~/Models/bartowski_cerebras_MiniMax-M2-REAP-162B-A10B-GGUF")))

   (make-llm-setup-model
    :name 'Phi-4-reasoning-plus
    :context-length 32768
    :temperature 0.6
    :min-p 0.01
    :top-p 0.9
    :supports-reasoning t
    :instances
    (list
     (make-llm-setup-instance
      :model-path "~/Models/unsloth_Phi-4-reasoning-plus-GGUF"
      :hostnames
      '("hera" "clio")
      :arguments
      '("--flash-attn" "on"))))

   (make-llm-setup-model
    :name 'Qwen3-30B-A3B
    :context-length 40000
    :temperature 0.2
    :min-p 0.0
    :top-p 0.9
    :top-k 20
    :supports-function-calling t
    :supports-reasoning nil
    :instances
    (list
     (make-llm-setup-instance
      :max-output-tokens 32000
      :model-path "~/Models/unsloth_Qwen3-30B-A3B-GGUF"
      :hostnames
      '("hera" "clio"))))

   (make-llm-setup-model
    :name 'Qwen3-Coder-Next
    :context-length 262144
    :temperature 1.0
    :min-p 0.01
    :top-p 0.9
    :top-k 40
    :supports-function-calling t
    :instances
    (list
     (make-llm-setup-instance
      :max-output-tokens 131072
      :model-path "~/Models/unsloth_Qwen3-Coder-Next-GGUF"
      :hostnames
      '("hera" "clio"))))

   (make-llm-setup-model
    :name 'Qwen3-Coder-Next-REAP-40B-A3B
    :context-length 262144
    :temperature 1.0
    :min-p 0.01
    :top-p 0.9
    :top-k 40
    :supports-function-calling t
    :instances
    (list
     (make-llm-setup-instance
      :max-output-tokens 131072
      :model-path "~/Models/mradermacher_Qwen3-Coder-Next-REAP-40B-A3B-GGUF"
      :hostnames
      '("hera" "clio"))))

   (make-llm-setup-model
    :name 'Qwen3.5-397B-A17B
    :context-length 262144
    :temperature 0.6
    :min-p 0.0
    :top-p 0.9
    :top-k 20
    :supports-function-calling t
    :supports-reasoning t
    :instances
    (list
     (make-llm-setup-instance
      :max-output-tokens 81920
      :model-path "~/Models/unsloth_Qwen3.5-397B-A17B-GGUF"
      :arguments '("--no-prefill-assistant")
      :cache-type-k 'q8_0)

     (make-llm-setup-instance
      :name 'mlx-community/Qwen3.5-397B-A17B-4bit
      :engine 'vllm-mlx)))

   (make-llm-setup-model
    :name 'Qwen3.5-397B-A17B-1M
    :context-length 1048576
    :temperature 0.6
    :min-p 0.0
    :top-p 0.9
    :top-k 20
    :supports-function-calling t
    :supports-reasoning t
    :instances
    (list
     (make-llm-setup-instance
      :max-output-tokens 81920
      :model-path "~/Models/unsloth_Qwen3.5-397B-A17B-GGUF"
      :arguments '("--no-prefill-assistant")
      :cache-type-k 'q8_0)))

   (make-llm-setup-model
    :name 'Qwen3.5-122B-A10B
    :context-length 262144
    :temperature 0.6
    :min-p 0.0
    :top-p 0.9
    :top-k 20
    :supports-function-calling t
    :supports-reasoning t
    :instances
    (list
     (make-llm-setup-instance
      :max-output-tokens 131072
      :model-path "~/Models/unsloth_Qwen3.5-122B-A10B-GGUF"
      :arguments '("--no-prefill-assistant")
      :cache-type-k 'q8_0
      :fallbacks '(clio/Qwen3.5-35B-A3B))

     (make-llm-setup-instance
      :name 'mlx-community/Qwen3.5-122B-A10B-4bit
      :engine 'vllm-mlx)))

   (make-llm-setup-model
    :name 'Qwen3.5-35B-A3B
    :context-length 262144
    :temperature 0.6
    :min-p 0.0
    :top-p 0.9
    :top-k 20
    :supports-function-calling t
    :supports-reasoning t
    :instances
    (list
     (make-llm-setup-instance
      :max-output-tokens 131072
      :model-path "~/Models/unsloth_Qwen3.5-35B-A3B-GGUF"
      :arguments '("--no-prefill-assistant")
      :cache-type-k 'q8_0
      :fallbacks '(clio/Qwen3.5-35B-A3B)
      :hostnames '("hera" "clio"))

     (make-llm-setup-instance
      :name 'mlx-community/Qwen3.5-35B-A3B-4bit
      :engine 'vllm-mlx)))

   (make-llm-setup-model
    :name 'Qwen3.5-27B
    :context-length 262144
    :temperature 0.6
    :min-p 0.0
    :top-p 0.9
    :top-k 20
    :supports-function-calling t
    :supports-reasoning t
    :instances
    (list
     (make-llm-setup-instance
      :max-output-tokens 131072
      :file-path "~/Models/unsloth_Qwen3.5-27B-GGUF/Qwen3.5-27B-UD-Q8_K_XL.gguf"
      :arguments '("--no-prefill-assistant")
      :parallel 1
      :cache-type-k 'q8_0
      :fallbacks '(clio/Qwen3.5-27B)
      :hostnames '("hera"))

     (make-llm-setup-instance
      :max-output-tokens 131072
      :file-path "~/Models/unsloth_Qwen3.5-27B-GGUF/Qwen3.5-27B-UD-Q4_K_XL.gguf"
      :arguments '("--no-prefill-assistant")
      :parallel 1
      :cache-type-k 'q8_0
      :fallbacks '(clio/Qwen3.5-27B)
      :hostnames '("clio"))

     (make-llm-setup-instance
      :name 'mlx-community/Qwen3.5-27B-4bit
      :engine 'vllm-mlx)))

   (make-llm-setup-model
    :name 'Qwen3.5-27B-Instruct
    :context-length 262144
    :temperature 0.7
    :min-p 0.0
    :top-p 0.8
    :top-k 20
    :supports-function-calling t
    :supports-reasoning nil
    :instances
    (list
     (make-llm-setup-instance
      :max-output-tokens 131072
      :file-path "~/Models/unsloth_Qwen3.5-27B-GGUF/Qwen3.5-27B-UD-Q4_K_XL.gguf"
      :parallel 1
      :cache-type-k 'q8_0
      :arguments
      '("--no-prefill-assistant"
        "--chat-template-kwargs"
        "'{\"enable_thinking\":false}'")
      :fallbacks '(clio/Qwen3.5-27B-Instruct)
      :hostnames '("hera" "clio"))))

   (make-llm-setup-model
    :name 'Qwen3.5-9B
    :context-length 262144
    :temperature 0.6
    :min-p 0.0
    :top-p 0.9
    :top-k 20
    :supports-function-calling t
    :supports-reasoning t
    :instances
    (list
     (make-llm-setup-instance
      :max-output-tokens 131072
      :model-path "~/Models/unsloth_Qwen3.5-9B-GGUF"
      :arguments '("--no-prefill-assistant")
      :cache-type-k 'q8_0
      :fallbacks '(clio/Qwen3.5-9B)
      :hostnames '("hera" "clio"))

     (make-llm-setup-instance
      :name 'mlx-community/Qwen3.5-9B-4bit
      :engine 'vllm-mlx)))

   (make-llm-setup-model
    :name 'Qwen3.5-9B-Instruct
    :context-length 262144
    :temperature 0.6
    :min-p 0.0
    :top-p 0.9
    :top-k 20
    :supports-function-calling t
    :supports-reasoning t
    :instances
    (list
     (make-llm-setup-instance
      :max-output-tokens 131072
      :model-path "~/Models/unsloth_Qwen3.5-9B-GGUF"
      :parallel 1
      :cache-type-k 'q8_0
      :arguments
      '("--no-prefill-assistant"
        "--chat-template-kwargs"
        "'{\"enable_thinking\":false}'")
      :fallbacks '(clio/Qwen3.5-9B-Instruct)
      :hostnames '("hera" "clio"))))

   (make-llm-setup-model
    :name 'Qwen3.5-4B
    :context-length 262144
    :temperature 0.6
    :min-p 0.0
    :top-p 0.9
    :top-k 20
    :supports-function-calling t
    :supports-reasoning t
    :instances
    (list
     (make-llm-setup-instance
      :max-output-tokens 131072
      :model-path "~/Models/unsloth_Qwen3.5-4B-GGUF"
      :arguments '("--no-prefill-assistant")
      :cache-type-k 'q8_0
      :fallbacks '(clio/Qwen3.5-4B)
      :hostnames '("hera" "clio"))

     (make-llm-setup-instance
      :name 'mlx-community/Qwen3.5-4B-4bit
      :engine 'vllm-mlx)))

   (make-llm-setup-model
    :name 'Qwen3.5-4B-Instruct
    :context-length 262144
    :temperature 0.6
    :min-p 0.0
    :top-p 0.9
    :top-k 20
    :supports-function-calling t
    :supports-reasoning nil
    :instances
    (list
     (make-llm-setup-instance
      :max-output-tokens 131072
      :model-path "~/Models/unsloth_Qwen3.5-4B-GGUF"
      :cache-type-k 'q8_0
      :arguments
      '("--no-prefill-assistant"
        "--chat-template-kwargs"
        "'{\"enable_thinking\":false}'")
      :fallbacks '(clio/Qwen3.5-4B)
      :hostnames '("hera" "clio"))))

   (make-llm-setup-model
    :name 'Qwen3.5-2B
    :context-length 262144
    :temperature 0.6
    :min-p 0.0
    :top-p 0.9
    :top-k 20
    :supports-function-calling t
    :supports-reasoning t
    :instances
    (list
     (make-llm-setup-instance
      :max-output-tokens 131072
      :model-path "~/Models/unsloth_Qwen3.5-2B-GGUF"
      :arguments '("--no-prefill-assistant")
      :cache-type-k 'q8_0
      :fallbacks '(clio/Qwen3.5-2B)
      :hostnames '("hera" "clio"))

     (make-llm-setup-instance
      :name 'mlx-community/Qwen3.5-2B-4bit
      :engine 'vllm-mlx)))

   (make-llm-setup-model
    :name 'Qwen3.5-2B-Instruct
    :context-length 262144
    :temperature 0.6
    :min-p 0.0
    :top-p 0.9
    :top-k 20
    :supports-function-calling t
    :supports-reasoning t
    :instances
    (list
     (make-llm-setup-instance
      :max-output-tokens 131072
      :model-path "~/Models/unsloth_Qwen3.5-2B-GGUF"
      :cache-type-k 'q8_0
      :arguments
      '("--no-prefill-assistant"
        "--chat-template-kwargs"
        "'{\"enable_thinking\":false}'")
      :fallbacks '(clio/Qwen3.5-2B)
      :hostnames '("hera" "clio"))))

   (make-llm-setup-model
    :name 'Qwen3.5-0.8B
    :context-length 262144
    :temperature 0.6
    :min-p 0.0
    :top-p 0.9
    :top-k 20
    :supports-function-calling t
    :supports-reasoning t
    :instances
    (list
     (make-llm-setup-instance
      :max-output-tokens 131072
      :model-path "~/Models/unsloth_Qwen3.5-0.8B-GGUF"
      :arguments '("--no-prefill-assistant")
      :cache-type-k 'q8_0
      :fallbacks '(clio/Qwen3.5-0.8B)
      :hostnames '("hera" "clio"))

     (make-llm-setup-instance
      :name 'mlx-community/Qwen3.5-0.8B-4bit
      :engine 'vllm-mlx)))

   (make-llm-setup-model
    :name 'gpt-oss-20b-MXFP4-Q8
    :context-length 131072
    :temperature 1.0
    :min-p 0.0
    :top-p 0.9
    :supports-function-calling t
    :supports-reasoning t
    :instances
    (list
     (make-llm-setup-instance
      :name 'mlx-community/gpt-oss-20b-MXFP4-Q8
      :hostnames
      '("hera" "clio")
      :engine 'mlx-lm)))

   (make-llm-setup-model
    :name 'gpt-oss-20b
    :context-length 131072
    :temperature 1.0
    :min-p 0.0
    :top-p 0.9
    :supports-function-calling t
    :supports-reasoning t
    :instances
    (list
     (make-llm-setup-instance
      :model-path "~/Models/unsloth_gpt-oss-20b-GGUF"
      :hostnames
      '("hera" "clio"))))

   (make-llm-setup-model
    :name 'gpt-oss-120b-MXFP4-Q8
    :context-length 131072
    :temperature 1.0
    :min-p 0.0
    :top-p 0.9
    :supports-function-calling t
    :supports-reasoning t
    :instances
    (list
     (make-llm-setup-instance
      :name 'mlx-community/gpt-oss-120b-MXFP4-Q8
      :engine 'mlx-lm)))

   (make-llm-setup-model
    :name 'gpt-oss-120b
    :context-length 131072
    :temperature 1.0
    :min-p 0.0
    :top-p 0.9
    :supports-function-calling t
    :supports-reasoning t
    :instances
    (list
     (make-llm-setup-instance
      :model-path "~/Models/unsloth_gpt-oss-120b-GGUF"
      ;; :draft-model "~/Models/unsloth_gpt-oss-20b-GGUF/gpt-oss-20b-Q8_0.gguf"
      ;; :fallbacks '(hera/claude-sonnet-4-5-20250929-thinking-32000
      ;;              anthropic/claude-sonnet-4-5-20250929)
      )))

   (make-llm-setup-model
    :name 'gpt-oss-safeguard-20b-MLX-MXFP4
    :context-length 131072
    :temperature 1.0
    :min-p 0.0
    :top-p 0.9
    :supports-function-calling t
    :supports-reasoning t
    :instances
    (list
     (make-llm-setup-instance
      :name 'lmstudio-community/gpt-oss-safeguard-20b-MLX-MXFP4
      :hostnames
      '("hera" "clio")
      :engine 'mlx-lm)))

   (make-llm-setup-model
    :name 'gpt-oss-safeguard-20b
    :context-length 131072
    :temperature 1.0
    :min-p 0.0
    :top-p 0.9
    :supports-function-calling t
    :supports-reasoning t
    :instances
    (list
     (make-llm-setup-instance
      :model-path "~/Models/unsloth_gpt-oss-safeguard-20b-GGUF")))

   (make-llm-setup-model
    :name 'Devstral-2-123B-Instruct-2512
    :context-length 262144
    :temperature 1.0
    :min-p 0.0
    :top-p 0.9
    :supports-function-calling t
    :instances
    (list
     (make-llm-setup-instance
      :model-path "~/Models/unsloth_Devstral-2-123B-Instruct-2512-GGUF")))

   (make-llm-setup-model
    :name 'Devstral-Small-2-24B-Instruct-2512
    :context-length 262144
    :temperature 1.0
    :min-p 0.0
    :top-p 0.9
    :supports-function-calling t
    :instances
    (list
     (make-llm-setup-instance
      :model-path "~/Models/unsloth_Devstral-Small-2-24B-Instruct-2512-GGUF")
     (make-llm-setup-instance
      :model-path "~/Models/unsloth_Devstral-Small-2-24B-Instruct-2512-GGUF"
      :hostnames '("clio")
      :context-length 140000)))

   (make-llm-setup-model
    :name 'Nemotron-3-Nano-30B-A3B
    :context-length 1048576
    :temperature 1.0
    :min-p 0.0
    :top-p 0.9
    :supports-function-calling t
    :instances
    (list
     (make-llm-setup-instance
      :model-path "~/Models/unsloth_Nemotron-3-Nano-30B-A3B-GGUF"
      :hostnames
      '("hera" "clio"))))

   (make-llm-setup-model
    :name 'Nemotron-Cascade-2-30B-A3B
    :context-length 262144
    :temperature 1.0
    :min-p 0.0
    :top-p 0.9
    :supports-function-calling t
    :instances
    (list
     (make-llm-setup-instance
      :model-path "~/Models/mradermacher_Nemotron-Cascade-2-30B-A3B-GGUF"
      :hostnames
      '("hera" "clio"))))

   (make-llm-setup-model
    :name 'NVIDIA-Nemotron-3-Super-120B-A12B
    :context-length 1048576
    :temperature 1.0
    :min-p 0.0
    :top-p 0.9
    :supports-function-calling t
    :instances
    (list
     (make-llm-setup-instance
      :model-path "~/Models/unsloth_NVIDIA-Nemotron-3-Super-120B-A12B-GGUF")))

   (make-llm-setup-model
    :name 'gemma-2-9b
    :context-length 131072
    :temperature 1.0
    :min-p 0.0
    :top-p 0.9
    :instances
    (list
     (make-llm-setup-instance
      :name 'google/gemma-2-9b
      :engine 'mlx-lm)))

   (make-llm-setup-model
    :name 'gemma-4-26B-A4B-it
    :context-length 131072
    :temperature 1.0
    :min-p 0.0
    :top-p 0.9
    :instances
    (list
     (make-llm-setup-instance
      :model-path "~/Models/unsloth_gemma-4-26B-A4B-it-GGUF"
      :hostnames
      '("hera" "clio"))))

   (make-llm-setup-model
    :name 'gemma-4-31B-it
    :context-length 131072
    :temperature 1.0
    :min-p 0.0
    :top-p 0.9
    :instances
    (list
     (make-llm-setup-instance
      :model-path "~/Models/unsloth_gemma-4-31B-it-GGUF"
      :hostnames
      '("hera" "clio"))))

   ;; (make-llm-setup-model
   ;;  :name 'Trinity-Large-Thinking
   ;;  :context-length 131072
   ;;  :temperature 1.0
   ;;  :min-p 0.0
   ;;  :top-p 0.9
   ;;  :instances
   ;;  (list
   ;;   (make-llm-setup-instance
   ;;    :model-path "~/Models/arcee-ai_Trinity-Large-Thinking-GGUF")))

   (make-llm-setup-model
    :name 'LFM2.5-350M
    :context-length 131072
    :temperature 1.0
    :min-p 0.0
    :top-p 0.9
    :instances
    (list
     (make-llm-setup-instance
      :model-path "~/Models/LiquidAI_LFM2.5-350M-GGUF"
      :hostnames
      '("hera" "clio"))))

   (make-llm-setup-model
    :name 'Bonsai-8B
    :context-length 131072
    :temperature 1.0
    :min-p 0.0
    :top-p 0.9
    :instances
    (list
     (make-llm-setup-instance
      :model-path "~/Models/prism-ml_Bonsai-8B-gguf"
      :hostnames
      '("hera" "clio"))))

   (make-llm-setup-model
    :name 'Seed-OSS-36B-Instruct
    :context-length 32768
    :temperature 1.0
    :min-p 0.0
    :top-p 0.9
    :instances
    (list
     (make-llm-setup-instance
      :name 'ByteDance-Seed/Seed-OSS-36B-Instruct
      :engine 'mlx-lm)))

   (make-llm-setup-model
    :name 'chinese-alpaca-2-7b
    :context-length 16384
    :temperature 1.0
    :min-p 0.0
    :top-p 0.9
    :instances
    (list
     (make-llm-setup-instance
      :name 'hfl/chinese-alpaca-2-7b
      :engine 'mlx-lm)))

   (make-llm-setup-model
    :name 'chatglm-6b
    :context-length 16384
    :temperature 1.0
    :min-p 0.0
    :top-p 0.9
    :instances
    (list
     (make-llm-setup-instance
      :name 'zai-org/chatglm-6b
      :engine 'mlx-lm)))

   (make-llm-setup-model
    :name 'chatglm2-6b
    :context-length 16384
    :temperature 1.0
    :min-p 0.0
    :top-p 0.9
    :instances
    (list
     (make-llm-setup-instance
      :name 'zai-org/chatglm2-6b
      :engine 'mlx-lm)))

   (make-llm-setup-model
    :name 'Llama-3.1-8B
    :context-length 131072
    :temperature 1.0
    :min-p 0.0
    :top-p 0.9
    :instances
    (list
     (make-llm-setup-instance
      :name 'meta-llama/Llama-3.1-8B
      :engine 'mlx-lm)))

   (make-llm-setup-model
    :name 'Llama-3.1-8B-Instruct
    :context-length 131072
    :temperature 1.0
    :min-p 0.0
    :top-p 0.9
    :instances
    (list
     (make-llm-setup-instance
      :name 'meta-llama/Llama-3.1-8B-Instruct
      :engine 'mlx-lm)))

   (make-llm-setup-model
    :name 'Llama-3.2-1B
    :context-length 131072
    :temperature 1.0
    :min-p 0.0
    :top-p 0.9
    :instances
    (list
     (make-llm-setup-instance
      :name 'meta-llama/Llama-3.2-1B
      :engine 'mlx-lm)))

   (make-llm-setup-model
    :name 'Mistral-7B-Instruct-v0.2
    :context-length 131072
    :temperature 1.0
    :min-p 0.0
    :top-p 0.9
    :instances
    (list
     (make-llm-setup-instance
      :name 'mistralai/Mistral-7B-Instruct-v0.2
      :engine 'mlx-lm)))

   (make-llm-setup-model
    :name 'Mistral-7B-Instruct-v0.3
    :context-length 131072
    :temperature 1.0
    :min-p 0.0
    :top-p 0.9
    :instances
    (list
     (make-llm-setup-instance
      :name 'mistralai/Mistral-7B-Instruct-v0.3
      :engine 'mlx-lm)))

   (make-llm-setup-model
    :name 'Mixtral-8x7B-Instruct-v0.1
    :context-length 131072
    :temperature 1.0
    :min-p 0.0
    :top-p 0.9
    :instances
    (list
     (make-llm-setup-instance
      :name 'mistralai/Mixtral-8x7B-Instruct-v0.1
      :engine 'mlx-lm)))

   (make-llm-setup-model
    :name 'Leanstral-2603
    :context-length 262144
    :temperature 1.0
    :min-p 0.0
    :top-p 0.9
    :supports-function-calling t
    :supports-reasoning t
    :instances
    (list
     (make-llm-setup-instance
      :model-path "~/Models/jackcloudman_Leanstral-2603-GGUF")))

   (make-llm-setup-model
    :name 'Qwen.Qwen3-Reranker-8B
    :kind 'reranker
    :instances
    (list
     (make-llm-setup-instance
      :model-path "~/Models/DevQuasar_Qwen.Qwen3-Reranker-8B-GGUF"
      :hostnames
      '("hera" "clio")
      :arguments
      '("--reranking" "--batch-size" "4096" "--ubatch-size" "2048"))))

   (make-llm-setup-model
    :name 'Qwen3-Embedding-8B
    :context-length 32767
    :kind 'embedding
    :instances
    (list
     (make-llm-setup-instance
      :model-path "~/Models/Qwen_Qwen3-Embedding-8B-GGUF"
      :hostnames '("hera" "clio")
      :arguments
      '("--embedding"
        "--pooling"
        "last"
        "--batch-size"
        "8192"
        "--ubatch-size"
        "2048"))))

   (make-llm-setup-model
    :name 'bge-m3
    :context-length 8192
    :kind 'embedding
    :instances
    (list
     (make-llm-setup-instance
      :model-path "~/Models/gpustack_bge-m3-GGUF"
      :hostnames '("hera" "clio")
      :arguments
      '("--embedding"
        "--pooling"
        "mean"
        "--batch-size"
        "8192"
        "--ubatch-size"
        "4096"))))

   (make-llm-setup-model
    :name 'bge-reranker-v2-m3
    :kind 'reranker
    :instances
    (list
     (make-llm-setup-instance
      :model-path "~/Models/gpustack_bge-reranker-v2-m3-GGUF"
      :hostnames
      '("hera" "clio")
      :arguments
      '("--reranking" "--batch-size" "8192" "--ubatch-size" "4096"))))

   (make-llm-setup-model
    :name 'nomic-embed-text-v2-moe
    :context-length 512
    :kind 'embedding
    :instances
    (list
     (make-llm-setup-instance
      :model-path "~/Models/nomic-ai_nomic-embed-text-v2-moe-GGUF"
      :hostnames '("hera" "clio")
      :arguments
      '("--embedding"
        "--pooling"
        "mean"
        "--batch-size"
        "8192"
        "--ubatch-size"
        "4096"))))

   (make-llm-setup-model
    :name 'bge-base-en-v1.5
    :kind 'embedding
    :instances
    (list
     (make-llm-setup-instance
      :name 'BAAI/bge-base-en-v1.5
      :engine 'mlx-lm)))

   (make-llm-setup-model
    :name 'bge-large-en-v1.5
    :kind 'embedding
    :instances
    (list
     (make-llm-setup-instance
      :name 'BAAI/bge-large-en-v1.5
      :engine 'mlx-lm
      :hostnames
      '("hera" "clio"))))

   (make-llm-setup-model
    :name 'cohere-transcribe-03-2026-mlx-8bit
    :instances
    (list
     (make-llm-setup-instance
      :name 'mlx-community/cohere-transcribe-03-2026-mlx-8bit
      :engine 'mlx-lm
      :hostnames
      '("hera" "clio"))))

   (make-llm-setup-model
    :name 'whisper-large-v3-mlx
    :instances
    (list
     (make-llm-setup-instance
      :name 'mlx-community/whisper-large-v3-mlx
      :engine 'mlx-lm)))

   (make-llm-setup-model
    :name 'NV-Embed-v2
    :kind 'embedding
    :instances
    (list
     (make-llm-setup-instance
      :name 'nvidia/NV-Embed-v2
      :engine 'mlx-lm)))

   (make-llm-setup-model
    :name 'all-MiniLM-L6-v2
    :kind 'embedding
    :instances
    (list
     (make-llm-setup-instance
      :name 'sentence-transformers/all-MiniLM-L6-v2
      :engine 'mlx-lm)))

   (make-llm-setup-model
    :name 'gemini-2.5-pro
    :description "Gemini 2.5 Pro (Positron)"
    :supports-function-calling t
    :instances
    (list (make-llm-setup-instance :provider 'positron_gemini)))

   (make-llm-setup-model
    :name 'gemini-3-pro-preview
    :description "Gemini 3 Pro (Positron)"
    :supports-function-calling t
    :instances
    (list (make-llm-setup-instance :provider 'positron_gemini)))

   (make-llm-setup-model
    :name 'gpt-5.3-codex
    :description "ChatGPT 5.3 Codex (Positron)"
    :supports-function-calling t
    :instances
    (list (make-llm-setup-instance :provider 'positron_openai)))

   (make-llm-setup-model
    :name 'gpt-5.2
    :description "ChatGPT 5.2 (Positron)"
    :supports-function-calling t
    :instances
    (list (make-llm-setup-instance :provider 'positron_openai)))

   (make-llm-setup-model
    :name 'gpt-5.2-codex
    :description "ChatGPT 5.2 Codex (Positron)"
    :supports-function-calling t
    :instances
    (list (make-llm-setup-instance :provider 'positron_openai)))

   (make-llm-setup-model
    :name 'gpt-5.1
    :description "ChatGPT model"
    :supports-function-calling t
    :instances
    (list
     (make-llm-setup-instance :provider 'positron_openai)

     (make-llm-setup-instance :provider 'openai)))

   (make-llm-setup-model
    :name 'claude-haiku
    :supports-function-calling t
    :instances
    (list
     (make-llm-setup-instance
      :model-name 'claude-haiku-4-5-20251001
      :name 'claude-haiku-4-5-20251001
      :provider 'vibe-proxy)

     (make-llm-setup-instance
      :name 'claude-haiku-4-5-20251001
      :provider 'positron_anthropic)

     (make-llm-setup-instance
      :name 'claude-haiku-4-5-20251001
      :provider 'anthropic)))

   (make-llm-setup-model
    :name 'claude-sonnet
    :supports-function-calling t
    :instances
    (list
     (make-llm-setup-instance
      :model-name 'claude-sonnet-4-6
      :name 'claude-sonnet-4-6-thinking-32000
      :provider 'vibe-proxy)

     (make-llm-setup-instance
      :model-name 'claude-sonnet-4-6
      :name 'claude-sonnet-4-6
      :provider 'vibe-proxy)

     (make-llm-setup-instance
      :name 'claude-sonnet-4-6
      :provider 'positron_anthropic)

     (make-llm-setup-instance
      :name 'claude-sonnet-4-6
      :provider 'anthropic)))

   (make-llm-setup-model
    :name 'claude-opus
    :supports-function-calling t
    :instances
    (list
     (make-llm-setup-instance
      :model-name 'claude-opus-4-6
      :name 'claude-opus-4-6-thinking-32000
      :provider 'vibe-proxy
      ;; :fallbacks '(hera/Qwen3.5-27B)
      :fallbacks '(hera/gpt-oss-120b))

     (make-llm-setup-instance
      :model-name 'claude-opus-4-6
      :name 'claude-opus-4-6
      :provider 'vibe-proxy
      ;; :fallbacks '(hera/Qwen3.5-27B-Instruct)
      :fallbacks '(hera/gpt-oss-120b))

     (make-llm-setup-instance
      :name 'claude-opus-4-6
      :provider 'positron_anthropic)

     (make-llm-setup-instance
      :name 'claude-opus-4-6
      :provider 'anthropic)))

   (make-llm-setup-model
    :name 'r1-1776
    :supports-reasoning t
    :instances
    (list (make-llm-setup-instance :provider 'perplexity)))

   (make-llm-setup-model
    :name 'sonar-deep-research
    :instances
    (list (make-llm-setup-instance :provider 'perplexity)))

   (make-llm-setup-model
    :name 'sonar-pro
    :instances
    (list (make-llm-setup-instance :provider 'perplexity)))

   (make-llm-setup-model
    :name 'sonar-reasoning-pro
    :supports-reasoning t
    :instances
    (list (make-llm-setup-instance :provider 'perplexity)))

   (make-llm-setup-model
    :name 'compound-beta
    :instances
    (list (make-llm-setup-instance :provider 'groq)))

   (make-llm-setup-model
    :name 'llama-3.3-70b
    :instances
    (list
     (make-llm-setup-instance
      :name 'llama-3.3-70b-instruct-good-tp2
      :provider 'positron)

     (make-llm-setup-instance
      :name 'llama-3.3-70b-versatile
      :provider 'groq)))

   (make-llm-setup-model
    :name 'Llama-Guard-4-12B
    :instances
    (list
     (make-llm-setup-instance
      :name 'meta-llama/Llama-Guard-4-12B
      :provider 'groq))))
  "List of configured models."
  :type '(repeat sexp)
  :group 'llm-setup)

(defun llm-setup-models-from-characteristics (&rest characteristics)
  "Return all models that provides the full list of CHARACTERISTICS."
  (cl-loop
   for model in llm-setup-models-list when
   (and-let* ((all-chars (llm-setup-model-characteristics model)))
     (cl-subsetp characteristics all-chars))
   return (llm-setup-model-name model)))

;; (llm-setup-models-from-characteristics 'high 'local 'thinking)

(defun llm-setup-make-models-hash ()
  "Build a hashtable from NAME to MODEL for `llm-setup-models-list'."
  (let ((h (make-hash-table)))
    (cl-loop
     for
     model
     in
     llm-setup-models-list
     for
     name
     =
     (llm-setup-model-name model)
     do
     (puthash name model h)
     finally
     (return h))))

(defun llm-setup-get-model (model-name &optional models-hash)
  "Using MODELS-HASH, find the model with the given MODEL-NAME."
  (let ((models-hash (or models-hash (llm-setup-make-models-hash))))
    (gethash model-name models-hash)))

(defun llm-setup-instances-list ()
  "Return list of all current instances."
  (cl-loop
   for model in llm-setup-models-list nconc
   (cl-loop
    for
    instance
    in
    (llm-setup-model-instances model)
    collect
    (cons model instance))))

(defun llm-setup-full-model-name (directory)
  "Based on a model DIRECTORY, return the canonical full model name."
  (let ((name (file-name-nondirectory directory)))
    (when (string-prefix-p "models--" name)
      (setq name (substring name 8)))
    (replace-regexp-in-string "--" "/" name)))

(defun llm-setup-short-model-name (model-name)
  "Given a full MODEL-NAME, return its short model name."
  (thread-last
   model-name
   file-name-nondirectory
   (replace-regexp-in-string "-gguf" "")
   (replace-regexp-in-string "-GGUF" "")
   (replace-regexp-in-string ".*_" "")))

(defconst llm-setup-gguf-min-file-size (* 5 1024 1024))

(defun llm-setup-get-gguf-path (model)
  "Find the best GGUF file for MODEL."
  (car
   (delete-dups
    (cl-loop
     for
     gguf
     in
     (sort (directory-files-recursively model "\\.gguf\\'") #'string<)
     nconc
     (cl-loop
      for (_name pattern) in
      '(("fp" "fp\\(16\\|32\\)[_-]")
        ("f" "[Ff]\\(16\\|32\\)")
        ("q" "[Qq][234568]_\\(.*XL\\)?"))
      when (string-match-p pattern gguf) when
      (> (file-attribute-size (file-attributes gguf))
         llm-setup-gguf-min-file-size)
      collect gguf)))))

(defun llm-setup-get-context-length (model)
  "Get maximum context length of MODEL."
  (when-let* ((gguf
               (expand-file-name (llm-setup-get-gguf-path model))))
    (with-temp-buffer
      (when (zerop (call-process "gguf-tools" nil t nil "show" gguf))
        (goto-char (point-min))
        (when (search-forward ".context_length" nil t)
          (when (re-search-forward "\\[uint32\\] \\([0-9]+\\)" nil t)
            (string-to-number (match-string 1))))))))

(defun llm-setup-http-get (endpoint)
  "GET request to ENDPOINT."
  (let ((url-request-method "GET")
        (url-request-extra-headers
         `(("Authorization" . ,(concat "Bearer " llm-setup-api-key))
           ("Content-Type" . "application/json"))))
    (with-current-buffer (url-retrieve-synchronously
                          (concat (llm-setup-api-base) endpoint) t)
      (goto-char (point-min))
      (re-search-forward "^$")
      (json-read))))

(defun llm-setup-get-models ()
  "Get list of available models from server."
  (condition-case err
      (let* ((response (llm-setup-http-get "/v1/models"))
             (data (alist-get 'data response))
             (models (make-hash-table :test 'equal)))
        (seq-doseq (item data)
          (let ((id (alist-get 'id item)))
            (puthash id (list "M") models)))
        models)
    (error
     (message "Error fetching models: %s" err)
     (make-hash-table :test 'equal))))

;; (inspect (llm-setup-get-models))

(defun llm-setup-download (entries)
  "Download models from HuggingFace ENTRIES."
  (interactive "sModel entries (space-separated): ")
  (dolist (entry (split-string entries))
    (let* ((parts (split-string entry "/"))
           (model
            (string-join (cl-subseq parts 0 (min 2 (length parts)))
                         "/"))
           (name (replace-regexp-in-string "/" "_" model)))
      (make-directory name t)
      (shell-command (format "git clone hf.co:%s %s" model name)))))

(defun llm-setup-checkout (models)
  "Checkout MODELS using Git LFS."
  (interactive "sModel files (space-separated): ")
  (dolist (model (split-string models))
    (when (file-regular-p model)
      (let ((dir (file-name-directory model))
            (base (file-name-nondirectory model)))
        (shell-command
         (format "cd %s && git lfs fetch --include %s" dir base))
        (shell-command
         (format "cd %s && git lfs checkout %s" dir base))
        (shell-command (format "cd %s && git lfs dedup" dir))))))

(defun llm-setup-import-lmstudio (models)
  "Import MODELS to LMStudio."
  (interactive "sModel files (space-separated): ")
  (dolist (model (split-string models))
    (when (file-regular-p model)
      (let* ((file-path (expand-file-name model))
             (base
              (replace-regexp-in-string
               (concat
                (regexp-quote llm-setup-gguf-models) "/")
               "" file-path))
             (name (replace-regexp-in-string "_" "/" base))
             (target
              (expand-file-name name llm-setup-lmstudio-models)))
        (make-directory (file-name-directory target) t)
        (when (file-exists-p target)
          (delete-file target))
        (add-name-to-file file-path target)))))

(defun llm-setup-import-ollama (models)
  "Import MODELS to Ollama."
  (interactive "sModel files (space-separated): ")
  (dolist (model (split-string models))
    (when (file-regular-p model)
      (let* ((file-path (expand-file-name model))
             (base-name (file-name-nondirectory model))
             (modelfile-name
              (replace-regexp-in-string
               "\\.gguf$" ".modelfile" base-name))
             (model-name
              (replace-regexp-in-string "\\.gguf$" "" base-name)))
        (with-temp-file modelfile-name
          (insert (format "FROM %s\n" file-path)))
        (shell-command
         (format "ollama create %s -f %s" model-name modelfile-name))
        (delete-file modelfile-name)))))

(defun llm-setup-show (model)
  "Show MODEL details."
  (interactive "sModel directory: ")
  (let ((gguf (llm-setup-get-gguf-path model)))
    (when gguf
      (shell-command (format "gguf-tools show %s" gguf)))))

(defun llm-setup-get-instance-model-name (model instance)
  "Return the model name for the given MODEL and INSTANCE."
  (or (llm-setup-instance-model-name instance)
      (llm-setup-model-name model)))

(defun llm-setup-get-instance-name (model instance)
  "Return the model name for the given MODEL and INSTANCE."
  (or (llm-setup-instance-name instance)
      (llm-setup-model-name model)))

(defun llm-setup-get-instance-context-length (model instance)
  "Find maximum context-length for the given MODEL and INSTANCE."
  (or (llm-setup-instance-context-length instance)
      (llm-setup-model-context-length model)))

(defun llm-setup-get-instance-max-input-tokens (model instance)
  "Find maximum output tokens for the given MODEL and INSTANCE."
  (or (llm-setup-instance-max-input-tokens instance)
      (llm-setup-model-max-input-tokens model)))

(defun llm-setup-get-instance-max-output-tokens (model instance)
  "Find maximum output tokens for the given MODEL and INSTANCE."
  (or (llm-setup-instance-max-output-tokens instance)
      (llm-setup-model-max-output-tokens model)))

(defun llm-setup-lookup-fallback-instance (fallback-name)
  "Look up the instance whose name matches FALLBACK-NAME.
Returns a cons cell (MODEL . INSTANCE) or nil if not found."
  (cl-loop
   for
   (model . instance)
   in
   (llm-setup-instances-list)
   when
   (eq fallback-name (llm-setup-get-instance-name model instance))
   return
   (cons model instance)))

(defun llm-setup-get-full-litellm-name (model instance)
  "Return the full LiteLLM model name for MODEL and INSTANCE.
For local/vibe-proxy providers, returns \"host/name\".
For remote providers, returns \"provider/name\"."
  (let ((provider (llm-setup-instance-provider instance))
        (name (llm-setup-get-instance-name model instance)))
    (if (memq provider '(local vibe-proxy))
        ;; For local instances, use the first hostname
        (format "%s/%s"
                (car (llm-setup-instance-hostnames instance))
                name)
      ;; For remote providers, use the provider name
      (format "%s/%s" provider name))))

(defun llm-setup-format-router-fallbacks ()
  "Collect all instance fallbacks and format as router_settings YAML.
Returns a string suitable for insertion into the LiteLLM config."
  (let ((fallback-entries nil))
    ;; Collect all fallback mappings
    (dolist (mi (llm-setup-instances-list))
      (cl-destructuring-bind
       (model . instance) mi
       (when-let* ((fallbacks (llm-setup-instance-fallbacks instance))
                   (provider (llm-setup-instance-provider instance)))
         (let ((hostnames
                (if (memq provider '(local vibe-proxy))
                    (llm-setup-instance-hostnames instance)
                  (list provider))))
           ;; For each host where this instance is available
           (dolist (host hostnames)
             (let*
                 ((source-name
                   (format "%s/%s"
                           host
                           (llm-setup-get-instance-name
                            model instance)))
                  ;; Resolve each fallback to its full name
                  ;; Fallbacks can be either:
                  ;; - Full names like 'openai/gpt-4.1 (already qualified)
                  ;; - Instance names like 'claude-sonnet-4-5-20250929-thinking-32000 (need lookup)
                  (resolved-fallbacks
                   (cl-loop
                    for
                    fb
                    in
                    fallbacks
                    for
                    fb-str
                    =
                    (symbol-name fb)
                    if
                    (string-match-p "/" fb-str)
                    ;; Already a full name, use as-is
                    collect
                    fb-str
                    else
                    ;; Look up the instance to get full name
                    for
                    fb-mi
                    =
                    (llm-setup-lookup-fallback-instance fb)
                    when
                    fb-mi
                    collect
                    (llm-setup-get-full-litellm-name
                     (car fb-mi) (cdr fb-mi)))))
               (when resolved-fallbacks
                 (push (cons source-name resolved-fallbacks)
                       fallback-entries))))))))
    ;; Format as YAML
    (if fallback-entries
        (concat
         "\n  fallbacks:\n"
         (mapconcat (lambda (entry)
                      (format "    - \"%s\": [%s]"
                              (car entry)
                              (mapconcat (lambda (fb)
                                           (format "\"%s\"" fb))
                                         (cdr entry)
                                         ", ")))
                    (nreverse fallback-entries)
                    "\n"))
      "")))

(defsubst llm-setup-remote-hostname-p (hostname)
  "Return non-nil if HOSTNAME is both non-nil and a remote host.
Remote means it does not match `llm-setup-default-hostname'."
  (and hostname (not (string= hostname llm-setup-default-hostname))))

(defsubst llm-setup-remote-path (path hostname)
  "Given a possibly remote HOSTNAME, return the correct PATH to reference it."
  (if (llm-setup-remote-hostname-p hostname)
      (concat "/ssh:" hostname ":" path)
    path))

(defun llm-setup-get-instance-gguf-path (instance &optional hostname)
  "Return file path for the GGUF file related to INSTANCE.
Optionally read the path on the given HOSTNAME."
  (or (llm-setup-instance-file-path instance)
      (when-let* ((path (llm-setup-instance-model-path instance)))
        (llm-setup-get-gguf-path
         (llm-setup-remote-path path hostname)))))

(defun llm-setup-strip-tramp-prefix (path)
  "Remove TRAMP protocol/host info from PATH, leaving only the remote part."
  (if (file-remote-p path)
      (file-remote-p path 'localname)
    path))

(defun llm-setup-insert-instance-llama-swap
    (model instance &optional hostname)
  "Instance the llama-swap.yaml config for MODEL and INSTANCE.
Optionally generate for the given HOSTNAME."
  (let* ((engine (llm-setup-instance-engine instance))
         (max-output-tokens
          (llm-setup-get-instance-max-output-tokens model instance))
         (context-length
          (llm-setup-get-instance-context-length model instance))
         (parallel (llm-setup-instance-parallel instance))
         (cache-type-k (llm-setup-instance-cache-type-k instance))
         (cache-type-v (llm-setup-instance-cache-type-v instance))
         (kv-offload (llm-setup-instance-kv-offload instance))
         (cache-prompt (llm-setup-instance-cache-prompt instance))
         (cache-reuse (llm-setup-instance-cache-reuse instance))
         (cache-ram (llm-setup-instance-cache-ram instance))
         (slot-save-path (llm-setup-instance-slot-save-path instance))
         (slot-prompt-similarity
          (llm-setup-instance-slot-prompt-similarity instance))
         (temperature (llm-setup-model-temperature model))
         (min-p (llm-setup-model-min-p model))
         (top-p (llm-setup-model-top-p model))
         (top-k (llm-setup-model-top-k model))
         (args
          (mapconcat
           #'identity
           (append
            (llm-setup-instance-arguments instance)
            (and temperature
                 (cl-case
                  engine
                  (vllm-mlx
                   (list
                    "--default-temperature"
                    (number-to-string temperature)))
                  (t (list "--temp" (number-to-string temperature)))))
            (and min-p
                 (not (eq engine 'vllm-mlx))
                 (list "--min-p" (number-to-string min-p)))
            (and top-p
                 (cl-case
                  engine
                  (vllm-mlx
                   (list "--default-top-p" (number-to-string top-p)))
                  (t (list "--top-p" (number-to-string top-p)))))
            (and top-k
                 (not (eq engine 'vllm-mlx))
                 (list "--top-k" (number-to-string top-k)))
            (and cache-type-k
                 (eq engine 'llama-cpp)
                 (list "--cache-type-k" (symbol-name cache-type-k)))
            (and cache-type-v
                 (eq engine 'llama-cpp)
                 (list "--cache-type-v" (symbol-name cache-type-v)))
            (and (not kv-offload)
                 (eq engine 'llama-cpp)
                 (list "--no-kv-offload"))
            (and (not cache-prompt)
                 (eq engine 'llama-cpp)
                 (list "--no-cache-prompt"))
            (and cache-reuse
                 (eq engine 'llama-cpp)
                 (list
                  "--cache-reuse" (number-to-string cache-reuse)))
            (and cache-ram
                 (eq engine 'llama-cpp)
                 (list "--cache-ram" (number-to-string cache-ram)))
            (and slot-save-path
                 (eq engine 'llama-cpp)
                 (list
                  "--slot-save-path"
                  (expand-file-name slot-save-path)))
            (and slot-prompt-similarity
                 (eq engine 'llama-cpp)
                 (list
                  "--slot-prompt-similarity"
                  (number-to-string slot-prompt-similarity)))
            (and-let* ((draft-model
                        (llm-setup-instance-draft-model instance))
                       (expanded (expand-file-name draft-model)))
              (and (file-exists-p expanded)
                   (cl-case
                    engine
                    (llama-cpp (list "--model-draft" expanded))
                    (mlx-lm (list "--draft-model" expanded)))))
            (and
             context-length
             (cl-case
              engine
              ;; mlx-lm and vllm-mlx do not specify the context size,
              ;; but grow the context dynamically based on usage.
              (llama-cpp
               (list
                "--ctx-size"
                (number-to-string (* context-length parallel))))))
            (and max-output-tokens
                 (list
                  (cl-case
                   engine
                   (llama-cpp "--predict")
                   ((mlx-lm vllm-mlx) "--max-tokens"))
                  (number-to-string max-output-tokens)))
            (and (eq engine 'vllm-mlx)
                 (> parallel 1)
                 (list
                  "--max-num-seqs"
                  (number-to-string parallel)
                  "--continuous-batching")))
           " "))
         (leader
          (format "
  \"%s\":
    proxy: \"http://127.0.0.1:${PORT}\"
    cmd: >"
                  (llm-setup-get-instance-name model instance)))
         (footer "
    checkEndpoint: /health
"))
    (cl-case
     engine
     (llama-cpp
      (when-let* ((path
                   (llm-setup-get-instance-gguf-path instance
                                                     hostname))
                  (exe
                   (let ((default-directory
                          (llm-setup-remote-path "~/" hostname)))
                     (executable-find
                      llm-setup-llama-server-executable
                      (llm-setup-remote-hostname-p hostname)))))
        (insert
         leader
         (format-spec
          "
      %e
        --host 127.0.0.1 --port ${PORT}
        --jinja
        --offline
        --parallel %n
        --model %p %a"
          `((?e . ,exe)
            (?p
             .
             ,(llm-setup-strip-tramp-prefix (expand-file-name path)))
            (?a . ,args) (?n . ,(number-to-string parallel))))
         footer)))
     (mlx-lm
      (when-let* ((exe
                   (let ((default-directory
                          (llm-setup-remote-path "~/" hostname)))
                     (executable-find llm-setup-mlx-lm-executable
                                      (llm-setup-remote-hostname-p
                                       hostname)))))
        (insert
         leader
         (format-spec
          "
      %e server
        --host 127.0.0.1 --port ${PORT}
        --use-default-chat-template
        --model %p %a"
          `((?e . ,exe)
            (?p . ,(llm-setup-get-instance-name model instance))
            (?a . ,args)))
         footer)))
     (vllm-mlx
      (when-let* ((exe
                   (let ((default-directory
                          (llm-setup-remote-path "~/" hostname)))
                     (executable-find llm-setup-vllm-mlx-executable
                                      (llm-setup-remote-hostname-p
                                       hostname)))))
        (insert
         leader
         (format-spec
          "
      %e serve %p
        --host 127.0.0.1 --port ${PORT} %a"
          `((?e . ,exe)
            (?p . ,(llm-setup-get-instance-name model instance))
            (?a . ,args)))
         footer))))))

(defun llm-setup-generate-llama-swap-yaml (hostname)
  "Build llama-swap.yaml configuration for HOSTNAME."
  (with-current-buffer (get-buffer-create "*llama-swap.yaml*")
    (erase-buffer)
    (insert llm-setup-llama-swap-prolog)
    (insert "\nmodels:")
    (dolist (mi (llm-setup-instances-list))
      (cl-destructuring-bind
       (model . instance) mi
       (when (and (memq
                   (llm-setup-instance-provider instance)
                   '(local vibe-proxy))
                  (member
                   hostname (llm-setup-instance-hostnames instance)))
         (llm-setup-insert-instance-llama-swap model instance
                                               hostname))))
    (insert llm-setup-llama-swap-epilog)
    (yaml-mode)
    (current-buffer)))

;; (display-buffer (llm-setup-generate-llama-swap-yaml "hera"))

(defun llm-setup-build-llama-swap-yaml (&optional hostname)
  "Build llama-swap.yaml configuration, optionally for HOSTNAME."
  (let* ((target-host (or hostname llm-setup-default-hostname))
         (yaml-path
          (if (string= hostname "vulcan")
              (expand-file-name "llama-swap.yaml"
                                "/home/johnw/Models")
            (expand-file-name "llama-swap.yaml"
                              llm-setup-gguf-models))))
    (message "[llama-swap] Generating YAML for %s..." target-host)
    (with-temp-buffer
      (insert
       (with-current-buffer (llm-setup-generate-llama-swap-yaml
                             target-host)
         (buffer-string)))
      (message "[llama-swap] Writing to %s..."
               (llm-setup-remote-path yaml-path hostname))
      (write-file (llm-setup-remote-path yaml-path hostname)))
    (message "[llama-swap] Stopping llama-swap on %s..." target-host)
    (if (and hostname
             (not (string= hostname llm-setup-default-hostname)))
        (shell-command
         (format "ssh %s killall llama-swap 2>/dev/null" hostname))
      (shell-command "killall llama-swap 2>/dev/null"))
    (message "[llama-swap] Done for %s" target-host)))

(defun llm-setup-insert-instance-litellm (model instance)
  "Instance the LiteLLM config for MODEL and INSTANCE."
  (let* ((hostnames (llm-setup-instance-hostnames instance))
         (provider (llm-setup-instance-provider instance))
         (cache-control (llm-setup-instance-cache-control instance))
         (_model-name
          (llm-setup-get-instance-model-name model instance))
         (name (llm-setup-get-instance-name model instance))
         (kind (llm-setup-model-kind model))
         (description (llm-setup-model-description model))
         (max-input-tokens
          (llm-setup-get-instance-max-input-tokens model instance))
         (max-output-tokens
          (llm-setup-get-instance-max-output-tokens model instance))
         (supports-system-message
          (llm-setup-model-supports-system-message model))
         (supports-function-calling
          (llm-setup-model-supports-function-calling model))
         (supports-reasoning
          (llm-setup-model-supports-reasoning model))
         (supports-response-schema
          (llm-setup-model-supports-response-schema model)))
    (dolist (host
             (if (memq provider '(local vibe-proxy))
                 hostnames
               (list provider)))
      (insert
       (format "
  - model_name: %s/%s
    litellm_params:
      model: %s/%s
      litellm_credential_name: %s_credential
      %ssupports_system_message: %s
    model_info:
      mode: %S
      description: %S%s%s
      supports_function_calling: %s
      supports_reasoning: %s
      supports_response_schema: %s
"
               host name
               (cond
                ((eq 'local provider)
                 "openai")
                ((eq 'vibe-proxy provider)
                 "openai")
                ((eq 'positron provider)
                 "openai")
                ((string-match
                  "positron_\\(.+\\)" (symbol-name provider))
                 (match-string 1 (symbol-name provider)))
                (t
                 provider))
               name
               (cond
                ((eq 'local provider)
                 (concat host "_llama_swap"))
                ((eq 'vibe-proxy provider)
                 (concat host "_vibe_proxy"))
                (t
                 provider))
               (if (eq kind 'embedding)
                   "drop_params: true
      encoding_format: \"float\"
      "
                 "")

               (concat
                (if supports-system-message
                    "true"
                  "false")
                (when cache-control
                  "
      cache_control_injection_points:
        - location: message
          role: system"))
               (if (or (null kind) (eq kind 'text-generation))
                   "chat"
                 kind)
               (or description "")
               (if max-input-tokens
                   (format "\n      max_input_tokens: %s"
                           max-input-tokens)
                 "")
               (if max-output-tokens
                   (format "\n      max_output_tokens: %s"
                           max-output-tokens)
                 "")
               (if supports-function-calling
                   "true"
                 "false")
               (if supports-reasoning
                   "true"
                 "false")
               (if supports-response-schema
                   "true"
                 "false"))))))

(defun llm-setup-generate-litellm-yaml ()
  "Build LiteLLM config.yaml configuration."
  (with-current-buffer (get-buffer-create "*litellm-config.yaml*")
    (erase-buffer)
    (insert llm-setup-litellm-prolog)
    (insert "model_list:")
    (dolist (mi (llm-setup-instances-list))
      (cl-destructuring-bind
       (model . instance)
       mi
       (llm-setup-insert-instance-litellm model instance)
       ;; (unless (string= model (downcase model))
       ;;   (llm-setup-insert-instance-litellm (downcase model) instance))
       ))
    (insert llm-setup-litellm-credentials)
    (insert (funcall llm-setup-litellm-environment-function))
    ;; Format the epilog with dynamically generated router fallbacks
    (insert
     (format llm-setup-litellm-epilog-spec
             (llm-setup-format-router-fallbacks)))
    (yaml-mode)
    (current-buffer)))

;; (display-buffer (llm-setup-generate-litellm-yaml))

(defun llm-setup-build-litellm-yaml ()
  "Build LiteLLM config.yaml configuration."
  (message "[litellm] Generating LiteLLM configuration...")
  (with-temp-buffer
    (insert
     (with-current-buffer (llm-setup-generate-litellm-yaml)
       (buffer-string)))
    (message "[litellm] Writing to %s..." llm-setup-litellm-path)
    (write-file llm-setup-litellm-path))
  (message "[litellm] Restarting LiteLLM service...")
  ;; (shell-command "ssh vulcan sudo systemctl restart litellm.service")
  (shell-command
   "sudo systemctl --user -M litellm@ restart litellm.service")
  (message "[litellm] Done"))

(defun llm-setup-reset ()
  "Reset all of the configuration files related to LLMs."
  (interactive)
  (message "[llm-setup-reset] Starting reset process...")
  ;; First check that everything is sane
  (message "[llm-setup-reset] Step 1/5: Checking instances...")
  (unless (= 0 (llm-setup-check-instances))
    (error "Failed to check installed and defined instances"))
  (message "[llm-setup-reset] Step 1/5: Instance check complete")
  ;; Update llama-swap configurations on all machines that run models
  (message
   "[llm-setup-reset] Step 2/5: Building llama-swap.yaml for %s..."
   llm-setup-default-hostname)
  (llm-setup-build-llama-swap-yaml)
  (message
   "[llm-setup-reset] Step 3/5: Building llama-swap.yaml for clio...")
  (llm-setup-build-llama-swap-yaml "clio")
  ;; (llm-setup-build-llama-swap-yaml "vulcan")
  ;; Update LiteLLM to refer to all local and remote models
  (message
   "[llm-setup-reset] Step 4/5: Building LiteLLM config and restarting service...")
  (llm-setup-build-litellm-yaml)
  ;; Update GPTel with instance list, to remain in sync with LiteLLM
  (message "[llm-setup-reset] Step 5/5: Updating GPTel backends...")
  (setq
   gptel-model llm-setup-default-instance-name
   gptel-backend (gptel-backends-make-litellm))
  (message "[llm-setup-reset] Complete!"))

(defun llm-setup-get-instance-gptel-backend
    (model instance &optional hostname)
  "Instance the llama-swap.yaml config for MODEL and INSTANCE.
If HOSTNAME is non-nil, only generate definitions for that host."
  (let* ((model-name (llm-setup-get-instance-name model instance)))
    (unless (memq (llm-setup-model-kind model) '(embedding reranker))
      (cl-loop
       for server in
       (let ((provider (llm-setup-instance-provider instance)))
         (if (or (null provider) (memq provider '(local vibe-proxy)))
             (llm-setup-instance-hostnames instance)
           (list provider)))
       when (or (null hostname) (string= server hostname)) collect
       (list
        (if hostname
            model-name
          (intern (format "%s/%s" server model-name)))
        :description (or (llm-setup-model-description model) "")
        :capabilities (llm-setup-model-capabilities model)
        :mime-types (llm-setup-model-mime-types model))))))

(defun llm-setup-gptel-backends (&optional hostname)
  "Return the GPTel backends for all defined instances.
If HOSTNAME is non-nil, only generate definitions for that host."
  (cl-loop
   for
   (model . instance)
   in
   (llm-setup-instances-list)
   for
   backends
   =
   (llm-setup-get-instance-gptel-backend model instance hostname)
   when
   backends
   nconc
   backends))

;; (inspect (llm-setup-gptel-backends))

(defun llm-setup-lookup-instance (model)
  "Return the instance whole model matches the symbol MODEL."
  (cl-loop
   for
   (m . instance)
   in
   (llm-setup-instances-list)
   when
   (eq model m)
   return
   instance))

(defun llm-setup-check-instances ()
  "Check all model and instances definitions."
  (interactive)
  (let ((models-hash (llm-setup-make-models-hash))
        (warnings 0)
        (host-count (length llm-setup-valid-hostnames))
        (host-idx 0))
    (message
     "[llm-setup-check] Scanning installed models on %d hosts..."
     host-count)
    (dolist (host llm-setup-valid-hostnames)
      (cl-incf host-idx)
      (message "[llm-setup-check]   Host %d/%d: Scanning %s..."
               host-idx
               host-count
               host)
      (let ((installed-models (llm-setup-installed-models host)))
        (message
         "[llm-setup-check]   Host %d/%d: Found %d models on %s"
         host-idx host-count (length installed-models) host)
        (dolist (installed installed-models)
          (unless (llm-setup-get-model installed models-hash)
            (warn "Missing model for host %s: %s" host installed)
            (cl-incf warnings))
          (unless (catch 'found
                    (dolist (mi (llm-setup-instances-list))
                      (cl-destructuring-bind
                       (model . instance) mi
                       (when (and (member
                                   host
                                   (llm-setup-instance-hostnames
                                    instance))
                                  (eq
                                   installed
                                   (llm-setup-model-name model)))
                         (throw 'found instance)))))
            (warn "Missing instance for host %s: %s" host installed)
            (cl-incf warnings)))))
    (message "[llm-setup-check] Validating %d model definitions..."
             (length llm-setup-models-list))
    (dolist (model llm-setup-models-list)
      (let ((capabilities (llm-setup-model-capabilities model))
            (mime-types (llm-setup-model-mime-types model))
            (kind (llm-setup-model-kind model)))
        (dolist (cap capabilities)
          (unless (member cap llm-setup-all-model-capabilities)
            (warn "Unknown capability: %S" cap)
            (cl-incf warnings)))
        (dolist (mime mime-types)
          (unless (member mime llm-setup-all-model-mime-types)
            (warn "Unknown mime-type: %S" mime)
            (cl-incf warnings)))
        (unless (member kind llm-setup-all-model-kinds)
          (warn "Unknown kind: %S" kind)
          (cl-incf warnings))))
    (let ((instances (llm-setup-instances-list)))
      (message
       "[llm-setup-check] Validating %d instance definitions..."
       (length instances))
      (dolist (mi instances)
        (cl-destructuring-bind
         (_model . instance) mi
         (let ((model-path (llm-setup-instance-model-path instance))
               (file-path (llm-setup-instance-file-path instance))
               (hostnames (llm-setup-instance-hostnames instance))
               (provider (llm-setup-instance-provider instance))
               (engine (llm-setup-instance-engine instance))
               (draft-model
                (llm-setup-instance-draft-model instance)))
           (unless (or (null model-path)
                       (file-directory-p model-path))
             (warn "Unknown model-path: %s" model-path)
             (cl-incf warnings))
           (unless (or (null file-path) (file-regular-p file-path))
             (warn "Unknown file-path: %s" file-path)
             (cl-incf warnings))
           (dolist (host hostnames)
             (unless (member host llm-setup-valid-hostnames)
               (warn "Unknown hostname: %s" host)
               (cl-incf warnings)))
           (unless (member provider llm-setup-all-model-providers)
             (warn "Unknown provider: %s" provider)
             (cl-incf warnings))
           (unless (member engine llm-setup-all-model-engines)
             (warn "Unknown engine: %s" engine)
             (cl-incf warnings))
           (unless (or (null draft-model)
                       (file-regular-p draft-model))
             (warn "Unknown draft-model: %S" draft-model)
             (cl-incf warnings))))))
    (message "[llm-setup-check] Validation complete: %d warning(s)"
             warnings)
    warnings))

(cl-defun
 llm-setup-run-mlx
 (model &key (port 8081))
 "Start mlx-lm with a specific MODEL on the given PORT."
 (interactive (list
               (read-string "Model: ")
               :port (read-number "Port: " 8081)))
 (let ((proc
        (start-process "mlx-lm" "*mlx-lm*" "mlx-lm"
                       "--model"
                       model
                       "--port"
                       (format "%d" port))))
   (set-process-query-on-exit-flag proc nil)
   (message "Started mlx-lm with model %s on port %d" model port)))

(cl-defun
 llm-setup-run-vllm-mlx
 (model &key (port 8081))
 "Start vllm-mlx with a specific MODEL on the given PORT."
 (interactive (list
               (read-string "Model: ")
               :port (read-number "Port: " 8081)))
 (let ((proc
        (start-process "vllm-mlx" "*vllm-mlx*" "vllm-mlx"
                       "serve"
                       model
                       "--port"
                       (format "%d" port))))
   (set-process-query-on-exit-flag proc nil)
   (message "Started vllm-mlx with model %s on port %d" model port)))

(cl-defun
 llm-setup-run-llama-cpp
 (model &key (port 8081))
 "Start llama.cpp with a specific MODEL on the given PORT."
 (interactive (list
               (read-string "Model: ")
               :port (read-number "Port: " 8081)))
 (let ((proc
        (start-process "llama-cpp" "*llama-cpp*" "llama-server"
                       "--jinja"
                       "--no-webui"
                       "--offline"
                       "--port"
                       (format "%d" port)
                       "--model"
                       model
                       "--threads"
                       (format "%d" llm-setup-threads))))
   (set-process-query-on-exit-flag proc nil)
   (message
    "Started llama.cpp with model %s on port %d using %d threads"
    model port llm-setup-threads)))

(defun llm-setup-run-llama-swap ()
  "Start llama-swap with generated config."
  (interactive)
  (let ((config-path
         (expand-file-name "llama-swap.yaml" llm-setup-gguf-models)))
    (shell-command (format "llama-swap --config %s" config-path))))

(defun llm-setup-status ()
  "Get llama-swap status."
  (interactive)
  (with-current-buffer (get-buffer-create "*Model Status*")
    (erase-buffer)
    (insert (json-encode (llm-setup-http-get "/running")))
    (json-pretty-print-buffer)
    (json-mode)
    (display-buffer (current-buffer))))

(defun llm-setup-unload ()
  "Unload current model."
  (interactive)
  (llm-setup-http-get "/unload")
  (message "Current model unloaded"))

(defun llm-setup-git-pull-all ()
  "Run git-pull in all model directories."
  (interactive)
  (async-shell-command
   (mapconcat
    (lambda (dir)
      (format "echo \">>> %s\" && cd \"%s\" && git pull" dir dir))
    (cl-loop
     for
     dir
     in
     (directory-files llm-setup-gguf-models t "\\`[^.]")
     when
     (file-directory-p dir)
     when
     (file-exists-p (expand-file-name ".git" dir))
     collect
     dir)
    " ; ")))

(defun llm-setup-installed-models (&optional hostname)
  "List all models from MLX and GGUF directories, optionally for HOSTNAME."
  (interactive)
  (cl-loop
   for
   base-dir
   in
   (list
    ;; (llm-setup-remote-path llm-setup-mlx-models hostname)
    (llm-setup-remote-path llm-setup-gguf-models hostname))
   do
   (message "[llm-setup-installed] Checking directory: %s" base-dir)
   when
   (file-exists-p base-dir)
   nconc
   (cl-loop
    for
    item
    in
    (directory-files base-dir t "\\`[^.]")
    when
    (file-directory-p item)
    unless
    (string= (file-name-nondirectory item) ".locks")
    collect
    (intern
     (llm-setup-short-model-name (llm-setup-full-model-name item))))))

;; (llm-setup-installed-models "vulcan")

(cl-defun
 llm-setup-generate-instance-declarations
 (&key (hostname llm-setup-default-hostname))
 "Generate model declarations from DIRECTORY's subdirectories.
These declarations are for HOSTNAME."
 (interactive)
 (let ((dirs
        (cl-remove-if-not
         #'file-directory-p
         (directory-files "~/Models" t "\\`[^.]"))))
   (with-current-buffer (get-buffer-create "*LLM-SETUP Instances*")
     (erase-buffer)
     (insert ";; Generated model configs from ~/Models\n")
     (dolist (dir dirs)
       (let* ((full-model (llm-setup-full-model-name dir))
              (model-name (llm-setup-short-model-name full-model))
              (context-length (llm-setup-get-context-length dir)))
         (insert
          "(make-llm-setup-model\n"
          "  :name '"
          model-name
          "\n"
          "  :context-length "
          (if (null context-length)
              "nil"
            (number-to-string context-length))
          "\n"
          "  :temperature 1.0\n"
          "  :min-p 0.05\n"
          "  :top-p 0.8\n"
          "  :top-k 20\n"
          "  :kind nil\n"
          "  :aliases '())\n\n")))
     (insert "\n;; Generated model instances from ~/Models\n")
     (dolist (dir dirs)
       (let* ((full-model (llm-setup-full-model-name dir))
              (model-name (llm-setup-short-model-name full-model)))
         (insert
          "(make-llm-setup-instance\n"
          "  :model '"
          model-name
          "\n"
          "  :hostnames '(\""
          hostname
          "\")\n"
          "  :file-format 'GGUF\n"
          "  :model-path \""
          dir
          "\"\n"
          "  :engine 'llama-cpp\n"
          "  :arguments '())\n\n")))
     (display-buffer (current-buffer)))))

(provide 'llm-setup)

;;; llm-setup.el ends here
