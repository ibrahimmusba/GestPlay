function varargout = soundGui(varargin)
% SOUNDGUI MATLAB code for soundGui.fig
%      SOUNDGUI, by itself, creates a new SOUNDGUI or raises the existing
%      singleton*.
%
%      H = SOUNDGUI returns the handle to a new SOUNDGUI or the handle to
%      the existing singleton*.
%
%      SOUNDGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SOUNDGUI.M with the given input arguments.
%
%      SOUNDGUI('Property','Value',...) creates a new SOUNDGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before soundGui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to soundGui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help soundGui

% Last Modified by GUIDE v2.5 25-Oct-2013 11:40:54

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @soundGui_OpeningFcn, ...
                   'gui_OutputFcn',  @soundGui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before soundGui is made visible.
function soundGui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to soundGui (see VARARGIN)

% Choose default command line output for soundGui
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes soundGui wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = soundGui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in chooseSoundFile.
function chooseSoundFile_Callback(hObject, eventdata, handles)
% hObject    handle to chooseSoundFile (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[file,path] = uigetfile('*.*','Select the file');
filename=[path file];
set(handles.filename, 'String',[path file]);



function NBegin_Callback(hObject, eventdata, handles)
% hObject    handle to NBegin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of NBegin as text
%        str2double(get(hObject,'String')) returns contents of NBegin as a double


% --- Executes during object creation, after setting all properties.
function NBegin_CreateFcn(hObject, eventdata, handles)
% hObject    handle to NBegin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function NEnds_Callback(hObject, eventdata, handles)
% hObject    handle to NEnds (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of NEnds as text
%        str2double(get(hObject,'String')) returns contents of NEnds as a double


% --- Executes during object creation, after setting all properties.
function NEnds_CreateFcn(hObject, eventdata, handles)
% hObject    handle to NEnds (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function filename_Callback(hObject, eventdata, handles)
% hObject    handle to filename (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of filename as text
%        str2double(get(hObject,'String')) returns contents of filename as a double


% --- Executes during object creation, after setting all properties.
function filename_CreateFcn(hObject, eventdata, handles)
% hObject    handle to filename (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



% --- Executes on button press in ReadSoundFile.
function ReadSoundFile_Callback(hObject, eventdata, handles)
% hObject    handle to ReadSoundFile (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
fileName  = get(handles.filename,'String');
N0 =  get(handles.NBegin,'String');
N1 = get(handles.NEnds,'String');
N(1) = str2num(N0);
if(N(1) >0) 
    N(1) = str2num(N0);
else
    N(1) =1;
end
N(2) = str2num(N1);

if ( N(2)<=0 && N(1) <=1)
[Y,Fs,NBITS,OPTS] = readMp3(fileName);
else 
[Y,Fs,NBITS,OPTS] = readMp3(fileName,N); 
end
handles.Y = Y;
handles.Fs = Fs;
guidata(hObject,handles);
display('File read!')
save('data.mat', 'Y', 'Fs','NBITS','OPTS');
 
% --- Executes on button press in PlaySound.
function PlaySound_Callback(hObject, eventdata, handles)
% hObject    handle to PlaySound (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Y = handles.Y;
Fs = handles.Fs;
sound(Y,Fs)
