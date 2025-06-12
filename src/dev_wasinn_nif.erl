-module(dev_wasinn_nif).
-export([load_http_server/2]).

-on_load(init/0).
-define(NOT_LOADED, not_loaded(?LINE)).

init() ->
    PrivDir = code:priv_dir(hb),
    NifPath = filename:join(PrivDir, "libdev_wasinn_nif"),
    erlang:load_nif(NifPath, 0).

not_loaded(Line) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, Line}]}).

load_http_server(_Port, _WasmModulePath) ->
    ?NOT_LOADED.