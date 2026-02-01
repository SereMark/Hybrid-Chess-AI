@echo off
setlocal EnableExtensions

set "FILENAME=thesis"
set "ENGINE=xelatex"

pushd "%~dp0" >nul

where %ENGINE% >nul 2>nul
if errorlevel 1 (
    echo [ERROR] LaTeX engine "%ENGINE%" not found.
    goto :end
)

echo [1/4] Deleting old auxiliary files: "%FILENAME%"...
for %%E in (
    aux log toc out lof lot
    bbl blg run.xml synctex.gz
    fls fdb_latexmk nav snm
    idx ilg ind
) do (
    if exist "%FILENAME%.%%E" del /q "%FILENAME%.%%E"
)
echo Done.
echo.

echo [2/4] Compilation 1 (Collecting citations)...
%ENGINE% -interaction=nonstopmode -file-line-error "%FILENAME%.tex"
if errorlevel 1 (
    set "FAILPASS=1"
    goto :fail
)
echo.

echo [3/4] Compilation 2 (Sorting cross-references)...
%ENGINE% -interaction=nonstopmode -file-line-error "%FILENAME%.tex" >nul
if errorlevel 1 (
    set "FAILPASS=2"
    goto :fail
)
echo.

echo [4/4] Compilation 3 (Final layout)...
%ENGINE% -interaction=nonstopmode -file-line-error "%FILENAME%.tex" >nul
if errorlevel 1 (
    set "FAILPASS=3"
    goto :fail
)

echo.
if exist "%FILENAME%.pdf" (
    echo      SUCCESS! "%FILENAME%.pdf" created.
) else (
    echo [WARNING] Compilation finished, but "%FILENAME%.pdf" not found.
)
echo.
goto :end

:fail
echo.
echo [ERROR] Compilation failed in pass %FAILPASS%.
echo.

:end
popd >nul
pause
endlocal
exit /b