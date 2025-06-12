%%% @doc This device offers an interface for validating AMD SEV-SNP commitments,
%%% as well as generating them, if called in an appropriate environment.
-module(dev_wasinn).
-export([load_http_server/2]).
-include("include/hb.hrl").

load_http_server(Port, WasmModuleName) ->
    PrivDir = code:priv_dir(hb),
    dev_wasinn_nif:load_http_server(Port, unicode:characters_to_binary(filename:join(PrivDir, WasmModuleName))).