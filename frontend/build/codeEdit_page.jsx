
import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  Code2, 
  Sparkles, 
  FileText, 
  Zap,
  Save,
  RotateCcw 
} from "lucide-react";
import CodeEditor from '../components/CodeEditor';

export default function CodeEditorDemo() {
  const [savedCode, setSavedCode] = useState('');
  const [lastSaved, setLastSaved] = useState(null);

  const sampleCode = `import React, { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Heart, Star } from "lucide-react";

export default function InteractiveCard() {
  const [likes, setLikes] = useState(0);
  const [isLiked, setIsLiked] = useState(false);

  const handleLike = () => {
    setIsLiked(!isLiked);
    setLikes(prev => isLiked ? prev - 1 : prev + 1);
  };

  return (
    <Card className="max-w-md mx-auto hover:shadow-lg transition-all duration-300">
      <CardContent className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-semibold">Live Code Editor</h3>
          <Badge variant="secondary" className="bg-blue-100 text-blue-800">
            <Star className="w-3 h-3 mr-1" />
            Interactive
          </Badge>
        </div>
        
        <p className="text-gray-600 mb-4">
          This card was created with the live code editor! 
          Try editing the code to see changes in real-time.
        </p>
        
        <div className="flex items-center justify-between">
          <Button 
            onClick={handleLike}
            variant={isLiked ? "default" : "outline"}
            className="flex items-center gap-2"
          >
            <Heart className={\`w-4 h-4 \\\${isLiked ? 'fill-current text-red-500' : ''}\`} />
            {likes} Likes
          </Button>
          
          <div className="text-sm text-gray-500">
            Click the floating code button to edit!
          </div>
        </div>
      </CardContent>
    </Card>
  );
}`;

  const handleSave = async (code) => {
    // Simulate save operation
    await new Promise(resolve => setTimeout(resolve, 1000));
    setSavedCode(code);
    setLastSaved(new Date().toLocaleTimeString());
  };

  const handleRevert = (originalCode) => {
    setSavedCode('');
    setLastSaved(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <div className="bg-white/80 backdrop-blur-sm border-b border-slate-200 sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                <Code2 className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Live Code Editor</h1>
                <p className="text-gray-600">Edit your frontend code in real-time</p>
              </div>
            </div>
            
            {lastSaved && (
              <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                <Save className="w-3 h-3 mr-1" />
                Saved at {lastSaved}
              </Badge>
            )}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Features Overview */}
          <div className="space-y-6">
            <Card className="border-none shadow-lg bg-white/80 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-yellow-500" />
                  Features
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
                    <FileText className="w-4 h-4 text-blue-600" />
                  </div>
                  <div>
                    <h3 className="font-semibold">Live Editing</h3>
                    <p className="text-sm text-gray-600">Edit your code with syntax highlighting and line numbers</p>
                  </div>
                </div>
                
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center">
                    <Save className="w-4 h-4 text-green-600" />
                  </div>
                  <div>
                    <h3 className="font-semibold">Save & Revert</h3>
                    <p className="text-sm text-gray-600">Save changes or revert back to original state</p>
                  </div>
                </div>
                
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 bg-purple-100 rounded-lg flex items-center justify-center">
                    <Zap className="w-4 h-4 text-purple-600" />
                  </div>
                  <div>
                    <h3 className="font-semibold">Keyboard Shortcuts</h3>
                    <p className="text-sm text-gray-600">Use Ctrl+S to save, Tab for indentation</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Instructions */}
            <Card className="border-none shadow-lg bg-gradient-to-r from-blue-50 to-purple-50">
              <CardHeader>
                <CardTitle className="text-blue-900">How to Use</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs font-bold">1</div>
                  <span>Click the floating code button in the bottom-right corner</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs font-bold">2</div>
                  <span>Edit the code in the full-featured editor</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs font-bold">3</div>
                  <span>Save your changes or revert to original</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs font-bold">4</div>
                  <span>Copy code or toggle fullscreen mode</span>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Live Preview Area */}
          <div className="space-y-6">
            <Card className="border-none shadow-lg bg-white">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Code2 className="w-5 h-5 text-purple-600" />
                  Live Preview
                </CardTitle>
              </CardHeader>
              <CardContent>
                {/* This would render the actual component from the edited code */}
                <div className="border-2 border-dashed border-gray-200 rounded-lg p-8 text-center">
                  <div className="max-w-md mx-auto">
                    <div className="w-16 h-16 bg-gradient-to-r from-blue-100 to-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
                      <Sparkles className="w-8 h-8 text-purple-600" />
                    </div>
                    <h3 className="text-xl font-semibold mb-2">Component Preview</h3>
                    <p className="text-gray-600 mb-4">
                      Your edited component will appear here in real-time
                    </p>
                    <Badge className="bg-blue-100 text-blue-800">
                      Live Updates
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            {savedCode && (
              <Card className="border-none shadow-lg bg-green-50">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-green-800">
                    <Save className="w-5 h-5" />
                    Saved Code
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <pre className="bg-green-100 p-4 rounded-lg text-sm overflow-auto max-h-40">
                    <code>{savedCode.slice(0, 200)}...</code>
                  </pre>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>

      {/* Code Editor Component */}
      <CodeEditor 
        initialCode={sampleCode}
        onSave={handleSave}
        onRevert={handleRevert}
        fileName="InteractiveCard.jsx"
      />
    </div>
  );
}
<DialogDescription className="text-sm text-slate-400 mt-1">
  Edit the code above to see live changes in the preview area.
</DialogDescription>