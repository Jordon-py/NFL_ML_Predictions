import React, { useState, useRef, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle,
  DialogFooter 
} from "@/components/ui/dialog";
import { 
  Code2, 
  Save, 
  RotateCcw, 
  X, 
  Copy, 
  Check,
  Maximize2,
  Minimize2
} from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";

export default function CodeEditor({ 
  initialCode = "// Welcome to the live code editor!\n// Edit your code here and see changes in real-time\n\nfunction MyComponent() {\n  return (\n    <div className=\"p-4\">\n      <h1>Hello, World!</h1>\n    </div>\n  );\n}\n\nexport default MyComponent;",
  onSave,
  onRevert,
  fileName = "Component.jsx"
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [currentCode, setCurrentCode] = useState(initialCode);
  const [originalCode] = useState(initialCode);
  const [hasChanges, setHasChanges] = useState(false);
  const [isCopied, setIsCopied] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [saveStatus, setSaveStatus] = useState(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    setHasChanges(currentCode !== originalCode);
  }, [currentCode, originalCode]);

  const handleSave = async () => {
    try {
      setSaveStatus('saving');
      if (onSave) {
        await onSave(currentCode);
      }
      setSaveStatus('success');
      setTimeout(() => setSaveStatus(null), 2000);
    } catch (error) {
      setSaveStatus('error');
      setTimeout(() => setSaveStatus(null), 3000);
    }
  };

  const handleRevert = () => {
    setCurrentCode(originalCode);
    if (onRevert) {
      onRevert(originalCode);
    }
    setSaveStatus('reverted');
    setTimeout(() => setSaveStatus(null), 2000);
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(currentCode);
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2000);
    } catch (error) {
      console.error('Failed to copy code:', error);
    }
  };

  const handleKeyDown = (e) => {
    // Handle Tab key for indentation
    if (e.key === 'Tab') {
      e.preventDefault();
      const textarea = textareaRef.current;
      const start = textarea.selectionStart;
      const end = textarea.selectionEnd;
      const newValue = currentCode.substring(0, start) + '  ' + currentCode.substring(end);
      setCurrentCode(newValue);
      
      // Set cursor position after the inserted spaces
      setTimeout(() => {
        textarea.selectionStart = textarea.selectionEnd = start + 2;
      }, 0);
    }
    
    // Handle Ctrl+S for save
    if ((e.ctrlKey || e.metaKey) && e.key === 's') {
      e.preventDefault();
      handleSave();
    }
  };

  const lineCount = currentCode.split('\n').length;

  return (
    <>
      {/* Floating Code Editor Button */}
      <Button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 z-40 h-14 w-14 rounded-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 shadow-lg hover:shadow-xl transition-all duration-300 group"
        size="icon"
      >
        <Code2 className="h-6 w-6 text-white group-hover:scale-110 transition-transform duration-200" />
      </Button>

      {/* Code Editor Modal */}
      <Dialog open={isOpen} onOpenChange={setIsOpen}>
        <DialogContent 
          className={`${
            isFullscreen 
              ? 'max-w-none w-screen h-screen m-0 rounded-none' 
              : 'max-w-6xl w-[95vw] h-[85vh]'
          } p-0 bg-slate-900 border-slate-700 transition-all duration-300`}
        >
          <DialogHeader className="px-6 py-4 border-b border-slate-700 bg-slate-800">
            <div className="flex items-center justify-between">
              <DialogTitle className="text-white text-lg font-semibold flex items-center gap-2">
                <Code2 className="h-5 w-5 text-blue-400" />
                {fileName}
                {hasChanges && <span className="text-orange-400 text-sm">●</span>}
              </DialogTitle>
              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setIsFullscreen(!isFullscreen)}
                  className="text-slate-400 hover:text-white hover:bg-slate-700"
                >
                  {isFullscreen ? (
                    <Minimize2 className="h-4 w-4" />
                  ) : (
                    <Maximize2 className="h-4 w-4" />
                  )}
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={handleCopy}
                  className="text-slate-400 hover:text-white hover:bg-slate-700"
                >
                  {isCopied ? (
                    <Check className="h-4 w-4 text-green-400" />
                  ) : (
                    <Copy className="h-4 w-4" />
                  )}
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setIsOpen(false)}
                  className="text-slate-400 hover:text-white hover:bg-slate-700"
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </DialogHeader>

          {/* Status Alert */}
          {saveStatus && (
            <div className="px-6 pt-4">
              <Alert 
                className={`${
                  saveStatus === 'success' || saveStatus === 'reverted' 
                    ? 'border-green-500 bg-green-500/10' 
                    : saveStatus === 'error' 
                    ? 'border-red-500 bg-red-500/10'
                    : 'border-blue-500 bg-blue-500/10'
                }`}
              >
                <AlertDescription className={`${
                  saveStatus === 'success' || saveStatus === 'reverted' 
                    ? 'text-green-400' 
                    : saveStatus === 'error' 
                    ? 'text-red-400'
                    : 'text-blue-400'
                }`}>
                  {saveStatus === 'saving' && 'Saving changes...'}
                  {saveStatus === 'success' && 'Changes saved successfully!'}
                  {saveStatus === 'error' && 'Failed to save changes. Please try again.'}
                  {saveStatus === 'reverted' && 'Code reverted to original state.'}
                </AlertDescription>
              </Alert>
            </div>
          )}

          {/* Code Editor */}
          <div className="flex-1 flex overflow-hidden">
            {/* Line Numbers */}
            <div className="bg-slate-800 px-3 py-4 border-r border-slate-700 select-none">
              <div className="font-mono text-sm text-slate-500 leading-6">
                {Array.from({ length: lineCount }, (_, i) => (
                  <div key={i + 1} className="text-right min-w-[2ch]">
                    {i + 1}
                  </div>
                ))}
              </div>
            </div>

            {/* Code Textarea */}
            <div className="flex-1 relative">
              <textarea
                ref={textareaRef}
                value={currentCode}
                onChange={(e) => setCurrentCode(e.target.value)}
                onKeyDown={handleKeyDown}
                className="w-full h-full p-4 bg-slate-900 text-white font-mono text-sm leading-6 resize-none outline-none border-none"
                placeholder="Start typing your code here..."
                spellCheck={false}
                style={{ 
                  tabSize: 2,
                  fontFamily: 'Monaco, Menlo, "Ubuntu Mono", monospace'
                }}
              />
            </div>
          </div>

          {/* Footer with Controls */}
          <DialogFooter className="px-6 py-4 border-t border-slate-700 bg-slate-800 flex justify-between items-center">
            <div className="flex items-center gap-4 text-sm text-slate-400">
              <span>Lines: {lineCount}</span>
              <span>Characters: {currentCode.length}</span>
              <span className="text-slate-500">Tip: Use Ctrl+S to save</span>
            </div>
            
            <div className="flex items-center gap-3">
              <Button
                variant="outline"
                onClick={handleRevert}
                disabled={!hasChanges}
                className="bg-slate-700 border-slate-600 text-white hover:bg-slate-600 disabled:opacity-50"
              >
                <RotateCcw className="h-4 w-4 mr-2" />
                Revert
              </Button>
              <Button
                onClick={handleSave}
                disabled={!hasChanges}
                className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:opacity-50"
              >
                <Save className="h-4 w-4 mr-2" />
                Save Changes
              </Button>
            </div>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}