// L2 Transport — public surface. See docs/lanes/L2-transport.md for the
// module map and the reasoning behind each piece; this file is deliberately
// thin (re-exports only) so that document, not this one, is where a reader
// goes for "why."

export type {
	ChainReader,
	GateFailureKind,
	GateFailure,
	GateOutcome,
	GateSuccess,
	OverrideRecord,
	StreamShapeKind,
	ClassifiedChunk,
	StreamShapeSummary,
	NormalizedError,
	ReceiptSnapshot
} from './types.js';

export {
	DEFAULT_WINDOW_N,
	MAX_WINDOW_N,
	assertValidWindowN,
	computeWindowIntent,
	verifyChainShape,
	runGate,
	applyOverride
} from './sendSet.js';

export {
	messageText,
	imageOnlyMarker,
	utf8SurrogatepassEncode,
	sha256Hex16,
	turnFingerprint,
	turnFingerprints,
	trailingAnchorFingerprints,
	ANCHOR_TURNS
} from './fingerprint.js';

export {
	mintConvId,
	assertSendableConvId,
	sanitizeConvId,
	wouldSanitizeToEmpty,
	buildRequestHeaders,
	buildChatCompletionBody
} from './request.js';
export type { ChatCompletionRequestParams } from './request.js';

export {
	STREAM_ID_PREFIX,
	classifyChunkId,
	classifyChunk,
	splitSseBlocks,
	extractDataPayload,
	parseSseEvent,
	parseErrorEnvelope,
	reduceStreamShape
} from './sse.js';
export type { SseParsedEvent } from './sse.js';

export { loadTransportConfig, DEFAULT_TIMEOUT_MS } from './config.js';
export type { TransportConfig } from './config.js';

export { streamChatCompletion, UpstreamHttpError, GenerationTimeoutError } from './transport.js';
export type { StreamChatCompletionParams } from './transport.js';

export { buildReceipt, readMessagesAdmitted, PROPOSED_ADMITTED_HEADER } from './receipt.js';
export type { HeaderReader, BuildReceiptParams } from './receipt.js';
