import {
  BaseChatModel,
  type BaseChatModelParams,
} from "@langchain/core/language_models/chat_models";
import {
  AIMessage,
  type BaseMessage,
  ChatMessage,
} from "@langchain/core/messages";
import { type ChatResult } from "@langchain/core/outputs";
import { type CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import { getEnvironmentVariable } from "@langchain/core/utils/env";

/**
 * Type representing the role of a message in the Deepseek chat model.
 */
export type DeepseekMessageRole = "system" | "assistant" | "user";

/**
 * Interface representing a message in the Deepseek chat model.
 */
interface DeepseekMessage {
  role: DeepseekMessageRole;
  content: string;
}



/**
 * Interface representing a request for a chat completion.
 */
interface ChatCompletionRequest {
  message: string;
  stream?: boolean;
  max_tokens?: number | null;
  top_p?: number | null;
  temperature?: number | null;
  frequency_penalty?: number | null;
  presence_penalty?: number | null;
}

/**
 * Interface representing a response from a chat completion.
 */
interface ChatCompletionResponse {
  code?: string;
  message?: string;
  request_id: string;
  usage: {
    output_tokens: number;
    input_tokens: number;
    total_tokens: number;
  };
  output: {
    text: string;
    finish_reason: "stop" | "length" | "null" | null;
  };
}

/**
 * Interface defining the input to the ChatDeepseek class.
 */
interface DeepseekChatInput {
  /**
   * Model name to use. Available options are: qwen-turbo, qwen-plus, qwen-max, or Other compatible models.
   * Alias for `model`
   * @default "qwen-turbo"
   */
  modelName: string;

  /** Model name to use. Available options are: qwen-turbo, qwen-plus, qwen-max, or Other compatible models.
   * @default "qwen-turbo"
   */
  model: string;

  /** Whether to stream the results or not. Defaults to false. */
  streaming?: boolean;

  /** Messages to pass as a prefix to the prompt */
  prefixMessages?: DeepseekMessage[];

  /**
   * API key to use when making requests. Defaults to the value of
   * `DEEPSEEK_API_KEY` environment variable.
   */
  deepseekApiKey?: string;

  /** Amount of randomness injected into the response. Ranges
   * from 0 to 1 (0 is not included). Use temp closer to 0 for analytical /
   * multiple choice, and temp closer to 1 for creative
   * and generative tasks. Defaults to 0.95.
   */
  temperature?: number;

  /** Total probability mass of tokens to consider at each step. Range
   * from 0 to 1.0. Defaults to 0.8.
   */
  topP?: number;

  maxTokens?: number;

  seed?: number;

  frequencyPenalty?: number | undefined;

  presencePenalty?: number | undefined;
}

/**
 * Function that extracts the custom role of a generic chat message.
 * @param message Chat message from which to extract the custom role.
 * @returns The custom role of the chat message.
 */
function extractGenericMessageCustomRole(message: ChatMessage) {
  if (["system", "assistant", "user"].includes(message.role) === false) {
    console.warn(`Unknown message role: ${message.role}`);
  }

  return message.role as DeepseekMessageRole;
}

/**
 * Function that converts a base message to a Deepseek message role.
 * @param message Base message to convert.
 * @returns The Deepseek message role.
 */
function messageToDeepseekRole(message: BaseMessage): DeepseekMessageRole {
  const type = message._getType();
  switch (type) {
    case "ai":
      return "assistant";
    case "human":
      return "user";
    case "system":
      return "system";
    case "function":
      throw new Error("Function messages not supported");
    case "generic": {
      if (!ChatMessage.isInstance(message)) { throw new Error("Invalid generic chat message"); }
      return extractGenericMessageCustomRole(message);
    }
    default:
      throw new Error(`Unknown message type: ${type}`);
  }
}

export class ChatDeepseek
  extends BaseChatModel
  implements DeepseekChatInput {
  static lc_name() {
    return "ChatDeepseek";
  }

  get callKeys() {
    return ["stop", "signal", "options"];
  }

  get lc_secrets() {
    return {
      deepseekApiKey: "DEEPSEEK_API_KEY",
    };
  }

  get lc_aliases() {
    return undefined;
  }

  lc_serializable = true;

  deepseekApiKey?: string;

  streaming: boolean;

  modelName = "deepseek-chat";

  model = "deepseek-chat";

  apiUrl: string;

  maxTokens?: number | undefined;

  temperature?: number | undefined;

  topP?: number | undefined;

  frequencyPenalty?: number | undefined;

  presencePenalty?: number | undefined;


  constructor(
    fields: Partial<DeepseekChatInput> & BaseChatModelParams = {}
  ) {
    super(fields);

    this.deepseekApiKey =
      fields?.deepseekApiKey ?? getEnvironmentVariable("DEEPSEEK_API_KEY");
    if (!this.deepseekApiKey) {
      throw new Error("Deepseek API key not found");
    }

    this.apiUrl = "https://chat.deepseek.com/api/v0/chat/completions";
    this.streaming = fields?.streaming ?? false;
    this.temperature = fields?.temperature;
    this.topP = fields?.topP;
    this.maxTokens = fields?.maxTokens;
    this.frequencyPenalty = fields?.frequencyPenalty;
    this.presencePenalty = fields?.presencePenalty;
    this.modelName = fields?.model ?? fields?.modelName ?? this.model;
    this.model = this.modelName;
  }

  /**
   * Get the parameters used to invoke the model
   */
  invocationParams(): Omit<ChatCompletionRequest, "message"> {
    return {
      stream: this.streaming,
      max_tokens: this.maxTokens,
      top_p: this.topP,
      temperature: this.temperature,
      frequency_penalty: this.frequencyPenalty,
      presence_penalty: this.presencePenalty,
    };
  }

  /**
   * Get the identifying parameters for the model
   */
  identifyingParams() {
    return {
      model: this.model,
      ... this.invocationParams(),
    };
  }

  /** @ignore */
  async _generate(
    messages: BaseMessage[],
    options?: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): Promise<ChatResult> {
    const parameters = this.invocationParams();

    const messagesMapped: string = messages.filter((message) => {
      return message._getType() === "human";
    })[0].content as string;

    const data = parameters.stream
      ? await new Promise<ChatCompletionResponse>((resolve, reject) => {
        let response: ChatCompletionResponse;
        let rejected = false;
        let resolved = false;
        this.completionWithRetry(
          {
            ...this.identifyingParams(),
            message: messagesMapped,
          },
          true,
          options?.signal,
          (event) => {
            const data: ChatCompletionResponse = JSON.parse(event.data);
            if (data?.code) {
              if (rejected) {
                return;
              }
              rejected = true;
              reject(new Error(data?.message));
              return;
            }

            const { text, finish_reason } = data.output;

            if (!response) {
              response = data;
            } else {
              response.output.text += text;
              response.output.finish_reason = finish_reason;
              response.usage = data.usage;
            }

            void runManager?.handleLLMNewToken(text ?? "");
            if (finish_reason && finish_reason !== "null") {
              if (resolved || rejected) {
                return;
              }
              resolved = true;
              resolve(response);
            }
          }
        ).catch((error) => {
          if (!rejected) {
            rejected = true;
            reject(error);
          }
        });
      })
      : await this.completionWithRetry(
        {
          ...this.identifyingParams(),
          message: messagesMapped,
        },
        false,
        options?.signal
      ).then<ChatCompletionResponse>((data) => {
        if (data?.code) {
          throw new Error(data?.message);
        }

        return data;
      });

    const {
      input_tokens = 0,
      output_tokens = 0,
      total_tokens = 0,
    } = data.usage;

    const { text } = data.output;

    return {
      generations: [
        {
          text,
          message: new AIMessage(text),
        },
      ],
      llmOutput: {
        tokenUsage: {
          promptTokens: input_tokens,
          completionTokens: output_tokens,
          totalTokens: total_tokens,
        },
      },
    };
  }

  /** @ignore */
  async completionWithRetry(
    request: ChatCompletionRequest,
    stream: boolean,
    signal?: AbortSignal,
    onmessage?: (event: MessageEvent) => void
  ) {
    const makeCompletionRequest = async () => {
      const response = await fetch(this.apiUrl, {
        method: "POST",
        headers: {
          ...(stream ? { Accept: "text/event-stream" } : {}),
          Authorization: `Bearer ${this.deepseekApiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(request),
        signal,
      });

      if (!stream) {
        return response.json();
      }

      if (response.body) {
        // response will not be a stream if an error occurred
        if (
          !response.headers.get("content-type")?.startsWith("text/event-stream")
        ) {
          onmessage?.(
            new MessageEvent("message", {
              data: await response.text(),
            })
          );
          return;
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let data = "";
        let continueReading = true;
        while (continueReading) {
          const { done, value } = await reader.read();
          if (done) {
            continueReading = false;
            break;
          }
          data += decoder.decode(value);
          let continueProcessing = true;
          while (continueProcessing) {
            const newlineIndex = data.indexOf("\n");
            if (newlineIndex === -1) {
              continueProcessing = false;
              break;
            }
            const line = data.slice(0, newlineIndex);
            data = data.slice(newlineIndex + 1);
            if (line.startsWith("data:")) {
              const event = new MessageEvent("message", {
                data: line.slice("data:".length).trim(),
              });
              onmessage?.(event);
            }
          }
        }
      }
    };

    return this.caller.call(makeCompletionRequest);
  }

  _llmType(): string {
    return "deepseek";
  }

  /** @ignore */
  _combineLLMOutput() {
    return [];
  }
}