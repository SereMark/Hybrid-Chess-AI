@echo off
setlocal EnableExtensions

set "FILENAME=thesis"
set "ENGINE=xelatex"

pushd "%~dp0" >nul

where %ENGINE% >nul 2>nul
if errorlevel 1 (
    echo [ERROR] LaTeX engine "%ENGINE%" nem található.
    goto :end
)

echo [1/4] Régi segédfájlok törlése: "%FILENAME%"...
for %%E in (
    aux log toc out lof lot
    bbl blg run.xml synctex.gz
    fls fdb_latexmk nav snm
    idx ilg ind
) do (
    if exist "%FILENAME%.%%E" del /q "%FILENAME%.%%E"
)
echo Kész.
echo.

echo [2/4] 1. fordítás (Hivatkozások gyűjtése)...
%ENGINE% -interaction=nonstopmode -file-line-error "%FILENAME%.tex"
if errorlevel 1 (
    set "FAILPASS=1"
    goto :fail
)
echo.

echo [3/4] 2. fordítás (Kereszthivatkozások rendezése)...
%ENGINE% -interaction=nonstopmode -file-line-error "%FILENAME%.tex" >nul
if errorlevel 1 (
    set "FAILPASS=2"
    goto :fail
)
echo.

echo [4/4] 3. fordítás (Végső tördelés)...
%ENGINE% -interaction=nonstopmode -file-line-error "%FILENAME%.tex" >nul
if errorlevel 1 (
    set "FAILPASS=3"
    goto :fail
)

echo.
if exist "%FILENAME%.pdf" (
    echo      SIKER! "%FILENAME%.pdf" létrehozva.
) else (
    echo [WARNING] A fordítás lefutott, de "%FILENAME%.pdf" nem található.
)
echo.
goto :end

:fail
echo.
echo [ERROR] A fordítás a(z) %FAILPASS%. passzban hibával leállt.
echo.

:end
popd >nul
pause
endlocal
exit /b