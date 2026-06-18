--[[
TagSelection.lua — Tague la SÉLECTION et écrit les mots-clés DANS le catalogue.

Flux complet :
  1. Dialogue natif Lightroom : choix des options (modèle, espèces, GPS…).
  2. Export de la sélection en JPEG + manifest.json (GPS, chemin, cible XMP).
  3. Lancement BLOQUANT du Python (selection_runner.py --no-questions …) :
     il calcule les tags et écrit results.json {tags_by_id: {uuid: [tags]}}.
  4. Lecture de results.json et écriture des mots-clés dans le catalogue via
     catalog:withWriteAccessDo (seul Lua peut écrire dans LrC en marche).

L'écriture catalogue est officielle et non destructive : les mots-clés
apparaissent immédiatement dans le panneau Mots-clés, photos encore
sélectionnées, sans réimport.
]]

local LrApplication   = import "LrApplication"
local LrTasks         = import "LrTasks"
local LrDialogs       = import "LrDialogs"
local LrExportSession = import "LrExportSession"
local LrPathUtils     = import "LrPathUtils"
local LrFileUtils     = import "LrFileUtils"
local LrDate          = import "LrDate"
local LrView          = import "LrView"
local LrBinding       = import "LrBinding"
local LrFunctionContext = import "LrFunctionContext"

-- >>> À ADAPTER si besoin <<<
local PROJECT_DIR = "/Users/laurentvoillot/Claude/photo-tagger"
local PYTHON_BIN  = PROJECT_DIR .. "/.venv/bin/python"
local RUNNER      = PROJECT_DIR .. "/selection_runner.py"

local function shellQuote(s)
    return "'" .. tostring(s):gsub("'", "'\\''") .. "'"
end

-- ---- JSON minimal (écriture) ----
local function jsonEscape(s)
    s = tostring(s)
    s = s:gsub("\\", "\\\\"):gsub('"', '\\"')
    s = s:gsub("\n", "\\n"):gsub("\r", "\\r"):gsub("\t", "\\t")
    return s
end
local function jsonValue(v)
    if v == nil then return "null"
    elseif type(v) == "boolean" then return v and "true" or "false"
    elseif type(v) == "number" then return string.format("%.8g", v)
    else return '"' .. jsonEscape(v) .. '"' end
end

-- ---- JSON minimal (lecture de results.json) ----
-- Format : { "tags_by_id": { "<id>": [ ["Lieu","Cambodge","Siem Reap"], ... ] } }
-- Chaque tag est un CHEMIN hiérarchique (tableau de niveaux). Un tag plat est
-- un chemin à un seul niveau (["temple"]).
local function parseResults(text)
    local result = {}
    local body = text:match('"tags_by_id"%s*:%s*(%b{})')
    if not body then return result end
    -- Pour chaque "id": [ ... ], on extrait la liste de chemins.
    for key, outer in body:gmatch('"([^"]+)"%s*:%s*(%b[])') do
        local paths = {}
        -- IMPORTANT : retirer les crochets externes du tableau de chemins, sinon
        -- %b[] re-capturerait tout le bloc au lieu de chaque sous-tableau.
        local inner = outer:sub(2, -2)
        -- Chaque chemin est un sous-tableau %b[].
        for pathArr in inner:gmatch('(%b[])') do
            local levels = {}
            for level in pathArr:gmatch('"([^"]*)"') do
                levels[#levels + 1] = level
            end
            if #levels > 0 then
                paths[#paths + 1] = levels
            end
        end
        result[key] = paths
    end
    return result
end

local function showOptionsDialog()
    return LrFunctionContext.callWithContext("phototagger_opts", function(context)
        local f = LrView.osFactory()
        local props = LrBinding.makePropertyTable(context)

        -- Liste des modèles ; le premier est le défaut sélectionné.
        local modelItems = {
            { title = "qwen3-vl:30b (qualité + animaux)", value = "qwen3-vl:30b" },
            { title = "qwen2.5vl:7b (rapide, FR)",        value = "qwen2.5vl:7b" },
        }
        props.model = modelItems[1].value  -- défaut = 1er de la liste
        props.species = false
        props.onlineSpecies = true
        props.onlinePlace = false
        props.writeXmp = false
        props.hierarchical = false  -- mots-clés hiérarchiques
        props.skipTagged = true  -- ignorer les photos déjà taguées IA
        props.suffix = "_AI"  -- suffixe par défaut, modifiable, peut être vide

        local c = f:column{
            bind_to_object = props,  -- résout les bindings dès la création (popup non vide)
            spacing = f:control_spacing(),
            f:static_text{ title = "Options de taggage de la sélection", font = "<system/bold>" },
            f:row{
                f:static_text{ title = "Modèle :", width = 110 },
                f:popup_menu{
                    value = LrView.bind("model"),
                    items = modelItems,
                    width = 280,
                },
            },
            f:row{
                f:static_text{ title = "Suffixe des tags :", width = 110 },
                f:edit_field{
                    value = LrView.bind("suffix"),
                    width = 120,
                    immediate = true,
                },
                f:static_text{ title = "(ex. _AI ; laisser vide pour aucun suffixe)" },
            },
            f:checkbox{ title = "Ignorer les photos déjà taguées par l'IA (même suffixe)", value = LrView.bind("skipTagged") },
            f:checkbox{ title = "Passe 2 espèces (BioCLIP, expérimental)", value = LrView.bind("species") },
            f:checkbox{ title = "Filtrer les espèces par GPS via GBIF (réseau)", value = LrView.bind("onlineSpecies") },
            f:checkbox{ title = "Enrichir les lieux via Nominatim/OSM (réseau)", value = LrView.bind("onlinePlace") },
            f:checkbox{ title = "Mots-clés hiérarchiques (Lieu>Pays>Ville, Faune>Classe>Espèce)", value = LrView.bind("hierarchical") },
            f:checkbox{ title = "Écrire aussi des sidecars .xmp", value = LrView.bind("writeXmp") },
        }

        local res = LrDialogs.presentModalDialog{
            title = "Photo Tagger — options",
            contents = c,
            actionVerb = "Lancer",
        }
        if res ~= "ok" then return nil end
        return {
            model = props.model,
            species = props.species,
            onlineSpecies = props.onlineSpecies,
            onlinePlace = props.onlinePlace,
            writeXmp = props.writeXmp,
            hierarchical = props.hierarchical,
            skipTagged = props.skipTagged,
            suffix = props.suffix or "",
        }
    end)
end

LrTasks.startAsyncTask(function()
    local catalog = LrApplication.activeCatalog()
    local photos = catalog:getTargetPhotos()
    if not photos or #photos == 0 then
        LrDialogs.message("Photo Tagger", "Aucune photo sélectionnée.", "warning")
        return
    end

    local opts = showOptionsDialog()
    if not opts then return end  -- annulé

    -- 0) Skip des photos déjà taguées par l'IA : on les retire AVANT l'export,
    --    pour ne pas exporter ni traiter inutilement (suffixe non vide requis).
    local nSkippedAlready = 0
    if opts.skipTagged and opts.suffix ~= "" then
        local suf = opts.suffix:lower()
        local kept = {}
        for _, photo in ipairs(photos) do
            local hasAI = false
            for _, kw in ipairs(photo:getRawMetadata("keywords") or {}) do
                local nm = kw:getName()
                if nm and nm:lower():sub(-#suf) == suf then
                    hasAI = true
                    break
                end
            end
            if hasAI then
                nSkippedAlready = nSkippedAlready + 1
            else
                kept[#kept + 1] = photo
            end
        end
        photos = kept
        if #photos == 0 then
            LrDialogs.message("Photo Tagger",
                "Toutes les photos sélectionnées (" .. nSkippedAlready ..
                ") sont déjà taguées par l'IA. Rien à faire.", "info")
            return
        end
    end

    -- 1) Dossier de travail temporaire.
    local stamp = LrDate.timeToUserFormat(LrDate.currentTime(), "%Y%m%d_%H%M%S")
    local workDir = LrPathUtils.child(LrPathUtils.getStandardFilePath("temp"),
                                      "phototagger_" .. stamp)
    LrFileUtils.createAllDirectories(workDir)

    -- 2) Export JPEG de la sélection.
    local exportSettings = {
        LR_export_destinationType = "specificFolder",
        LR_export_destinationPathPrefix = workDir,
        LR_export_useSubfolder = false,
        LR_format = "JPEG",
        LR_jpeg_quality = 0.8,
        LR_size_doConstrain = true,
        LR_size_maxHeight = 2048,
        LR_size_maxWidth = 2048,
        LR_size_resolution = 240,
        LR_collisionHandling = "rename",
        LR_includeVideoFiles = false,
        LR_embeddedMetadataOption = "all",
        LR_removeLocationMetadata = false,
        LR_renamingTokensOn = true,
        LR_tokens = "{{naming_sequenceNumber_5Digit}}",
        LR_tokenCustomString = "",
    }
    local session = LrExportSession({ photosToExport = photos, exportSettings = exportSettings })

    local jpegByLocalId = {}
    session:doExportOnCurrentTask()
    for _, rendition in session:renditions() do
        local ok, path = rendition:waitForRender()
        if ok then
            jpegByLocalId[rendition.photo.localIdentifier] = LrPathUtils.leafName(path)
        end
    end

    -- 3) Manifeste. IMPORTANT : on identifie chaque photo par son localIdentifier
    --    (UNIQUE par copie), et NON par l'uuid d'image — car les copies virtuelles
    --    et copies empilées partagent le même uuid, ce qui ferait collisionner
    --    leurs tags. On garde une table id -> photo pour l'écriture finale.
    local photoById = {}
    local parts = { "{\n", '  "catalog": ' .. jsonValue(catalog:getPath()) .. ",\n", '  "photos": [\n' }
    local n = 0
    for _, photo in ipairs(photos) do
        local jpeg = jpegByLocalId[photo.localIdentifier]
        if jpeg then
            n = n + 1
            local id = tostring(photo.localIdentifier)  -- unique par copie
            local origPath = photo:getRawMetadata("path")
            -- "fileName" n'est PAS une clé raw metadata valide -> on dérive le nom
            -- depuis le chemin (toujours disponible en raw).
            local fileName = LrPathUtils.leafName(origPath)
            local folder = LrPathUtils.parent(origPath)
            local base = LrPathUtils.removeExtension(fileName)
            local xmpPath = LrPathUtils.child(folder, base .. ".xmp")
            local gps = photo:getRawMetadata("gps")
            local lat, lon, hasGps = nil, nil, false
            if gps and gps.latitude and gps.longitude then
                lat, lon, hasGps = gps.latitude, gps.longitude, true
            end
            photoById[id] = photo
            local sep = (n > 1) and ",\n" or ""
            parts[#parts + 1] = sep .. "    {"
                .. '"id": '      .. jsonValue(id) .. ", "
                .. '"file": '    .. jsonValue(jpeg) .. ", "
                .. '"name": '    .. jsonValue(fileName) .. ", "
                .. '"folder": '  .. jsonValue(folder) .. ", "
                .. '"xmp": '     .. jsonValue(xmpPath) .. ", "
                .. '"lat": '     .. jsonValue(lat) .. ", "
                .. '"lon": '     .. jsonValue(lon) .. ", "
                .. '"has_gps": ' .. jsonValue(hasGps) .. "}"
        end
    end
    parts[#parts + 1] = "\n  ]\n}\n"
    local manifestPath = LrPathUtils.child(workDir, "manifest.json")
    local mf = io.open(manifestPath, "w"); mf:write(table.concat(parts)); mf:close()

    -- 4) Lancement NON BLOQUANT du Python (en arrière-plan) pour pouvoir afficher
    --    l'avancement pendant le traitement. Un fichier sentinelle "done.flag"
    --    contenant le code retour signale la fin ; "progress.json" donne l'état.
    local stdoutPath = LrPathUtils.child(workDir, "python_stdout.txt")
    local donePath   = LrPathUtils.child(workDir, "done.flag")
    local progPath   = LrPathUtils.child(workDir, "progress.json")

    local pycmd = shellQuote(PYTHON_BIN) .. " " .. shellQuote(RUNNER)
        .. " " .. shellQuote(manifestPath) .. " --no-questions"
        .. " --model " .. shellQuote(opts.model)
        .. " --suffix " .. shellQuote(opts.suffix or "_AI")
    if opts.species        then pycmd = pycmd .. " --species" end
    if not opts.onlineSpecies then pycmd = pycmd .. " --no-online-species" end
    if opts.onlinePlace    then pycmd = pycmd .. " --online-place" end
    if opts.writeXmp       then pycmd = pycmd .. " --write-xmp" end
    if opts.hierarchical   then pycmd = pycmd .. " --hierarchical" end

    -- sous-shell détaché : lance le Python, mémorise son PID (pour pouvoir le
    -- tuer en cas d'annulation), puis écrit son code retour dans done.flag.
    local pidPath = LrPathUtils.child(workDir, "python.pid")
    local shell = "( " .. pycmd .. " > " .. shellQuote(stdoutPath) .. " 2>&1 & "
        .. "PID=$! ; echo $PID > " .. shellQuote(pidPath) .. " ; "
        .. "wait $PID ; echo $? > " .. shellQuote(donePath) .. " ) &"
    LrTasks.execute("/bin/sh -c " .. shellQuote(shell))

    -- Lit la progression écrite par Python (done/total/current).
    local function readProgress()
        if not LrFileUtils.exists(progPath) then return nil end
        local pf = io.open(progPath, "r"); if not pf then return nil end
        local t = pf:read("*a"); pf:close()
        local done = tonumber(t:match('"done"%s*:%s*(%d+)'))
        local total = tonumber(t:match('"total"%s*:%s*(%d+)'))
        local current = t:match('"current"%s*:%s*"([^"]*)"')
        return done, total, current
    end

    -- Fenêtre de progression native Lightroom (barre en bas + annulation).
    local rc = 0
    LrFunctionContext.callWithContext("phototagger_progress", function(ctx)
        local progress = LrDialogs.showModalProgressDialog{
            title = "Photo Tagger — taggage en cours",
            caption = "Préparation… (le 1er chargement du modèle peut être long)",
            cannotCancel = false,
            functionContext = ctx,
        }
        while true do
            if progress:isCanceled() then
                -- Tue le process Python, mais on taguera quand même les photos
                -- déjà traitées (results.json est écrit au fil de l'eau).
                if LrFileUtils.exists(pidPath) then
                    local pf = io.open(pidPath, "r")
                    local pid = pf and pf:read("*a")
                    if pf then pf:close() end
                    if pid then
                        pid = pid:gsub("%s+", "")
                        if pid ~= "" then
                            -- SIGTERM, puis SIGKILL en repli si le process est
                            -- bloqué (ex. socket en lecture vers Ollama).
                            LrTasks.execute("/bin/kill " .. pid .. " 2>/dev/null")
                            LrTasks.sleep(0.5)
                            LrTasks.execute("/bin/kill -9 " .. pid .. " 2>/dev/null")
                        end
                    end
                end
                rc = -2  -- code interne : annulé, mais on tague le partiel
                break
            end
            if LrFileUtils.exists(donePath) then
                local df = io.open(donePath, "r")
                rc = tonumber((df:read("*a")) or "0") or 0
                df:close()
                progress:setPortionComplete(1, 1)
                break
            end
            local done, total, current = readProgress()
            if done and total and total > 0 then
                progress:setPortionComplete(done, total)
                progress:setCaption(string.format("Photo %d / %d : %s",
                    done, total, current or ""))
            end
            LrTasks.sleep(0.5)
        end
        progress:done()
    end)

    -- rc == -2 : annulé par l'utilisateur, mais on continue pour taguer les
    -- photos DÉJÀ traitées (results.json partiel). rc > 0 : vraie erreur.
    local wasCanceled = (rc == -2)
    if rc ~= 0 and not wasCanceled then
        LrDialogs.message("Photo Tagger",
            "Le script Python a échoué (code " .. tostring(rc) .. ").\n"
            .. "Voir python_stdout.txt dans :\n" .. workDir, "critical")
        return
    end

    -- 5) Lecture des résultats et écriture des mots-clés DANS le catalogue.
    --    Après une annulation, on laisse un court instant le process se terminer
    --    et finir d'écrire results.json.
    if wasCanceled then LrTasks.sleep(1.0) end
    local resultsPath = LrPathUtils.child(workDir, "results.json")
    if not LrFileUtils.exists(resultsPath) then
        LrDialogs.message("Photo Tagger",
            wasCanceled and "Annulé : aucune photo n'avait encore été taguée."
                or ("results.json introuvable.\n" .. workDir),
            wasCanceled and "info" or "critical")
        return
    end
    local rf = io.open(resultsPath, "r"); local rtext = rf:read("*a"); rf:close()
    local tagsById = parseResults(rtext)

    local suffix = opts.suffix or ""

    -- Base d'un mot-clé = sa forme SANS le suffixe (pour la déduplication).
    -- Ainsi "herbe" et "herbe_AI" ont la même base "herbe".
    local function baseOf(name)
        local lower = name:lower()
        if suffix ~= "" then
            local s = suffix:lower()
            if #lower > #s and lower:sub(-#s) == s then
                return lower:sub(1, #lower - #s)
            end
        end
        return lower
    end

    -- Crée (ou retrouve) le mot-clé feuille d'un chemin hiérarchique, en créant
    -- au passage chaque niveau parent. Le suffixe n'est appliqué qu'à la feuille.
    -- createKeyword(name, synonyms, includeOnExport, parent, returnExisting).
    local function resolvePath(levels)
        local parent = nil
        local leafKw = nil
        for i, level in ipairs(levels) do
            local isLeaf = (i == #levels)
            local name = isLeaf and (level .. suffix) or level
            leafKw = catalog:createKeyword(name, {}, true, parent, true)
            if not leafKw then return nil end
            parent = leafKw
        end
        return leafKw
    end

    local nPhotos, nTags, nSkipped = 0, 0, 0
    catalog:withWriteAccessDo("Ajout des mots-clés IA", function()
        for id, paths in pairs(tagsById) do
            local photo = photoById[id]
            if photo then
                nPhotos = nPhotos + 1

                -- Feuilles déjà présentes sur la photo, indexées par base.
                local existing = {}
                for _, kw in ipairs(photo:getRawMetadata("keywords") or {}) do
                    existing[baseOf(kw:getName())] = true
                end

                for _, levels in ipairs(paths) do
                    local leaf = levels[#levels]
                    local base = baseOf(leaf)  -- dédup sur la feuille
                    if existing[base] then
                        nSkipped = nSkipped + 1
                    else
                        local kw = resolvePath(levels)
                        if kw then
                            photo:addKeyword(kw)
                            nTags = nTags + 1
                            existing[base] = true
                        end
                    end
                end
            end
        end
    end)

    LrDialogs.message("Photo Tagger",
        (wasCanceled and "Traitement ANNULÉ — seules les photos déjà traitées "
            .. "ont été taguées.\n\n" or "Mots-clés écrits dans le catalogue.\n")
        .. nPhotos .. " photo(s), " .. nTags .. " mot(s)-clé(s) ajouté(s)"
        .. ((nSkipped > 0) and (", " .. nSkipped .. " ignoré(s) (déjà présents)") or "")
        .. ((nSkippedAlready > 0) and ("\n" .. nSkippedAlready
            .. " photo(s) déjà taguée(s) IA, non retraitée(s).") or "")
        .. ".\n\n"
        .. "Rapport : " .. workDir, "info")
end)
