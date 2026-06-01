--[[
Info.lua — Manifeste du plugin Lightroom Classic « Photo Tagger ».

Ajoute une entrée dans le menu Module externe > Extras qui tague les photos
SÉLECTIONNÉES (et non le catalogue entier) via le pipeline Python local.
]]

return {
    LrSdkVersion = 12.0,
    LrSdkMinimumVersion = 10.0,
    LrToolkitIdentifier = "com.laurentvoillot.phototagger",
    LrPluginName = "Photo Tagger (IA locale)",

    LrExportMenuItems = {
        {
            title = "Taguer la sélection avec l'IA locale…",
            file = "TagSelection.lua",
        },
    },

    LrLibraryMenuItems = {
        {
            title = "Taguer la sélection avec l'IA locale…",
            file = "TagSelection.lua",
        },
    },

    VERSION = { major = 0, minor = 1, revision = 0 },
}
