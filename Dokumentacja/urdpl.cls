%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% File:     urdpl.cls           (LaTeX Document class option "urdpl")
%          
% Author:  Zbigniew Gomółka     (zgomolka@ur.edu.pl)
%          Ewa Żesławska        (ezeslawska@ur.edu.pl)
%          Uniwersystet Rzeszowski, Rzeszów, POLAND
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\message{*** [urdpl.cls] ---  (ZG, EZ)  v1.0 <19.09.2024>  ***}

\newif\ifpdflatex\pdflatexfalse
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{urdpl}
\DeclareOption{pdflatex}{\pdflatextrue}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
% \ProcessOptions\LoadClass[a4paper,twoside,openright]{report} % openright- każdy nowy rozdział zaczyna się na prawej stronie
\ProcessOptions\LoadClass[a4paper,twoside]{report}

% Poniższe pakiety zapewniają pełną kontrolę nad typografią, formatowaniem tekstu, obrazów, kolorów i strukturą dokumentu.

\RequirePackage{times}
\RequirePackage[T1]{fontenc}    % sposób kodowania czcionek polskie znaki ą ś ć itp
\RequirePackage{graphicx}       % obsługa grafiki w dokumentach latez
\RequirePackage{geometry}
\RequirePackage[usenames]{color}
\RequirePackage{fancyhdr}       % zaawansowanego formatowania nagłówków i stopek stron.
\RequirePackage{url}            % zapewnia poprawne wyświetlanie i łamanie długich adresów URL w dokumencie.
\RequirePackage{titlesec}       % dostosowanie wyglądu tytułów rozdziałów i sekcji
\RequirePackage{tocloft}        % dostowanie wcięć, ostępów i innych, precyzyjne kontrolowanie wyglądu i układu spisu treści, srysunków w dokumencie

\RequirePackage[font=small, labelfont=bf, labelsep=period]{caption}     % zaawansowane formatowanie podpisów pod obrazkami, tabelami itp 
% font=small: Ustawia mniejszy rozmiar czcionki dla podpisu.
% labelfont=bf: Ustawia pogrubioną czcionkę dla etykiety podpisu (np. "Rysunek 1.").
% labelsep=period: Używa kropki po etykiecie, np. "Rysunek 1." zamiast "Rysunek 1:".
\RequirePackage{subcaption}     % obsługa subcaption czyli a) b) c) itp 
\RequirePackage{siunitx}        % wstawiania jednostek fizycznych i liczb z formatowaniem zgodnym z układem SI
\RequirePackage{enumerate}      % rozszerzenie do środowiska enumerate można robić listy numerowane różne np I. A. itp
\RequirePackage{enumitem}       % zaawansowane narzędzia do kontrolowania odstępów i formatowania w listach numerowanych
\RequirePackage[all,defaultlines=2]{nowidow} % Kontrola sierot i wdów - działa lepiej niż \widowpenalty i \clubpenalty

% Define geometry (margins)
\geometry{lmargin=35mm, rmargin=25mm, tmargin=25mm, bmargin=25mm, foot=10mm, head=10mm}
\definecolor{Green}{rgb}{0.1,0.5,0.1}

\def\shorttitlePL#1   {\gdef\@shorttitlePL{#1}}             % krótki tytuł pracy PL
\def\shorttitleEN#1   {\gdef\@shorttitleEN{#1}}             % krótki tytuł pracy EN
\def\shortauthor#1    {\gdef\@shortauthor{#1}}              % krótka nazwa autora 
\def\titlePL#1        {\gdef\@titlePL{#1}}                  % tytuł pracy PL
\def\titleEN#1        {\gdef\@titleEN{#1}}                  % tytuł pracy EN
\def\faculty#1      {\gdef\@faculty{#1}}                    % Nazwa Kolegium
\def\thesistype#1   {\gdef\@thesistype{#1}}                 % przechowuej informacje o pracy inżynierka magisterska
\def\supervisor#1   {\gdef\@supervisor{#1}}                 % nazwisko promotora
\def\department#1   {\gdef\@department{#1}}                 % Nazwa instytutu lub katerdy
\def\degreeprogramme#1 {\gdef\@degreeprogramme{#1}}         % nazwa kierunku studiów 
\def\noAlbum#1 {\gdef\@noAlbum{#1}}                         % numer albumu
\def\thesisDone#1 {\gdef\@thesisDone{#1}}                         % numer albumu

\linespread{1.3}                    % odstęp miedzy wierszami 1.5 w latex 1.3 poniewaz zapis 1.3 interpretowany jest jako 1.5
%\renewcommand{\labelitemi}{--}


%------------------------------------------------------------------------------
% polecenie do stron parzyste nieparzyste wyśweitlanie autora oraz którkiego tytułu pracy 
\def\thesisheaders{
    \fancyfoot[LE,RO]{\small \@shortauthor\quad\textit{\@shorttitlePL}}
}


\ProcessOptions

\newcommand{\titlepages}{%

\newpage \titlepage
\noindent
\begin{minipage}{0.6\textwidth}
    \raggedright
    {\sffamily{\large{\bfseries{\scshape UNIWERSYTET RZESZOWSKI}}}
        \vspace*{1mm}

        {\small\bfseries \MakeUppercase{\@faculty}} \newline
        {\small\bfseries \MakeUppercase{\@department}}}
    \vspace*{8mm}
\end{minipage}
\hfill
\begin{minipage}{0.4\textwidth}
    \raggedleft
    \vspace*{0mm}
    \includegraphics[scale=0.7]{figures/ur}
    \vspace*{2mm}
\end{minipage}

\begin{center}
    \vspace*{30mm}
    {\sffamily\Large\itshape \@author}
    \vspace*{2mm}

    {\sffamily\Large \@noAlbum}
    \vspace*{9mm}

    {\sffamily\Large\itshape \@degreeprogramme}
    \vspace*{20mm}

    {\sffamily\Large\itshape \@titlePL}
    \vspace*{15mm}

    {\raggedleft \sffamily\@thesistype \par}
    \vspace*{9mm}

\end{center}
\vspace{30mm}
{\raggedleft  \sffamily \@thesisDone \par}
\hspace{100mm}{\sffamily \@supervisor}
\vspace*{9mm}

\vfill
\centerline{\sffamily Rzesz\'{o}w \@date}


\clearpage \titlepage


\clearpage
\thispagestyle{empty}
\vspace*{15cm} \vfill
\clearpage

\setcounter{page}{5}

}


\pagestyle{fancy}
\renewcommand{\chaptermark}[1]{\markboth{\thechapter.\ #1}{}}
\renewcommand{\sectionmark}[1]{\markright{\thesection.\ #1}}
\renewcommand{\headrulewidth}{0.5pt}
\renewcommand{\footrulewidth}{0.5pt}
\fancyhf{}
\fancyhead{}
%\fancyhead[L]{\slshape{\small \rightmark}}
\fancyhead[RE,LO]{\slshape{\small \rightmark}}
%\fancyhead[R]{\bfseries \thepage}
\fancyhead[RO,LE]{\bfseries \thepage}

\thesisheaders

\frenchspacing
\sloppy
\flushbottom


%---------------------------------------------------------------------------
% Ustawienia parametrów tytułów rozdziałów, podrozdziałów, itd.
%---------------------------------------------------------------------------

\titleformat{\chapter}[block]
{\bfseries\LARGE}{\filright \LARGE\thechapter. }{0ex}{}
%odstępy: lewy, góra, dół
\titlespacing{\chapter}{0pt}{0mm}{8mm}
\titleformat{\section}[block]
{\bfseries\Large}{\filright \Large\thesection. }{0ex}{}
\titlespacing{\section}{7mm}{3mm plus 0mm minus 1mm}{3mm plus 0mm minus 1mm}
\titleformat{\subsection}[block]
{\bfseries\large}{\filright \large\thesubsection. }{0ex}{}
\titlespacing{\subsection}{7mm}{3mm}{2.5mm}
\titleformat{\subsubsection}[block]
{\bfseries\normalsize}{\filright \normalsize\thesubsubsection. }{0ex}{}
\titlespacing{\subsubsection}{7mm}{3mm plus 0mm minus 1mm}{3mm plus 0mm minus 1mm}

%---------------------------------------------------------------------------
% Ustawienia parametrów generowania spisu treści i spisu rysunków
%---------------------------------------------------------------------------

\renewcommand{\cfttoctitlefont}{\bfseries\Large}
\renewcommand{\cftbeforetoctitleskip}{0mm}
\renewcommand{\cftaftertoctitleskip}{10mm}
\renewcommand{\cftchapleader}{\cftdotfill{\cftdot}}
\renewcommand{\cftsecleader}{\cftdotfill{\cftdot}}
\renewcommand{\cftsubsecleader}{\cftdotfill{\cftdot}}
\renewcommand{\cftchapaftersnum}{.}
\renewcommand{\cftsecaftersnum}{.}
\renewcommand{\cftsubsecaftersnum}{.}
\setlength{\cftparskip}{2pt}
\setlength{\cftbeforechapskip}{0mm}
\renewcommand{\cftchappagefont}{\normalsize}



%---------------------------------------------------------------------------
% Ustawienia wyświetlania liczb (zgodne z polskimi zwyczajami typograficznymi)
%---------------------------------------------------------------------------

\sisetup{
    output-decimal-marker = {,},
    %	round-mode=places,
    %	round-precision=4,
    group-separator={~},
}


%---------------------------------------------------------------------------
% Ustawienia wyświetlania URL-i
%---------------------------------------------------------------------------

% Zmiana kroju czcionki URL-i z czcionki o stałej szerokości na szeryfową z kursywą.
\renewcommand\UrlFont{\rmfamily\itshape}

%---------------------------------------------------------------------------
% Ustawienia podpisów pod ilustracjami
%---------------------------------------------------------------------------

\captionsetup[subfigure]{labelfont=md}
\captionsetup{%
    % Użyj okrągłych nawiasów wokół odwołań do "podilustracji".
    subrefformat=parens,
    % Margines z obu stron podpisu.
    margin=2cm,
    % W przypadku podpisów mieszczących się w jednej linii nie stosuj justowania, tylko wyśrodkuj cały podpis.
    singlelinecheck=on,
}

%---------------------------------------------------------------------------
% Środowisko do tworzenia opisów symboli użytych w równaniu.
%---------------------------------------------------------------------------

\SetLabelAlign{parright}{\parbox[t]{\labelwidth}{\raggedleft#1}}
\setlist[description]{style=multiline, topsep=10pt, align=parright}

% Odstęp po obu stronach półpauzy.
\newlength{\descriptiondashgap}
\setlength{\descriptiondashgap}{0.5em}

% Element na liście symboli matematycznych użytych w równaniu (symbol wyrównany do prawej, oddzielony półpauzą od definicji)
\renewcommand*\descriptionlabel[1]{\hspace \labelsep \normalfont \bfseries #1 \mdseries\hspace{\descriptiondashgap}--\hspace{\descriptiondashgap}}

% Odległość półpauzy od lewego marginesu.
\newlength{\eqwheremargin}
\setlength{\eqwheremargin}{2cm}

\newenvironment{eqwhere}[1][]
{%
    \setlength{\eqwheremargin}{#1}%
    \begin{description}[leftmargin=\eqwheremargin, itemsep=0cm, labelsep=0cm]
        }
        {\end{description}}

